package com.example.facelocaldetect

import com.google.mlkit.vision.face.FaceLandmark
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.facelocaldetect.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var executor: ExecutorService

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else Toast.makeText(this, "需要相机权限", Toast.LENGTH_SHORT).show()
    }

    private val db by lazy { LocalDB(this) }
    private var persons: MutableList<Person> = mutableListOf()

    // 注册采样
    private var isRegistering = false
    private var registerName: String = ""
    private val tempSamples = mutableListOf<FloatArray>()

    // ML Kit 人脸检测器
    private val detector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .enableTracking()
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()
        FaceDetection.getClient(options)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        executor = Executors.newSingleThreadExecutor()
        persons = db.loadAll().toMutableList()

        // 注册按钮
        findViewById<Button>(R.id.btnRegister).setOnClickListener {
            if (isRegistering) {
                Toast.makeText(this, "正在采集，请稍等…", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val input = EditText(this).apply { hint = "输入姓名" }
            AlertDialog.Builder(this)
                .setTitle("注册人脸")
                .setView(input)
                .setPositiveButton("开始") { _, _ ->
                    val name = input.text.toString().trim()
                    if (name.isEmpty()) {
                        Toast.makeText(this, "姓名不能为空", Toast.LENGTH_SHORT).show()
                        return@setPositiveButton
                    }
                    registerName = name
                    tempSamples.clear()
                    isRegistering = true
                    Toast.makeText(this, "面对相机左右转头，采集 5 张", Toast.LENGTH_LONG).show()
                }
                .setNegativeButton("取消", null)
                .show()
        }

        // 清空库
        findViewById<Button>(R.id.btnClear).setOnClickListener {
            persons.clear()
            db.clear()
            Toast.makeText(this, "已清空向量库", Toast.LENGTH_SHORT).show()
        }

        // 权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        detector.close()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

            val selector = CameraSelector.DEFAULT_FRONT_CAMERA

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(executor) { imageProxy: ImageProxy ->
                processFrame(imageProxy)
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, selector, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image ?: run { imageProxy.close(); return }
        val rotation = imageProxy.imageInfo.rotationDegrees
        val input = InputImage.fromMediaImage(mediaImage, rotation)

        detector.process(input)
            .addOnSuccessListener { faces: List<Face> ->
                // 数据喂给 Overlay：传入检测到的人脸与原始帧尺寸
                val srcW = mediaImage.width
                val srcH = mediaImage.height

                val infos = faces.map { face ->
                    val emb = landmarkEmbedding(face)  // 16 维简单向量
                    var label: String? = null

                    if (emb != null) {
                        if (isRegistering) {
                            if (tempSamples.size < 5) {
                                tempSamples.add(emb)
                                if (tempSamples.size == 5) {
                                    persons.add(Person(registerName, tempSamples.toList()))
                                    db.saveAll(persons)
                                    isRegistering = false
                                }
                            }
                            label = "采集中: ${tempSamples.size}/5"
                        } else {
                            // 识别（与本地库余弦相似度最高者）
                            var best = -1f
                            var bestName: String? = null
                            for (p in persons) {
                                var top = -1f
                                for (vec in p.vectors) {
                                    val s = MathUtil.cosine(emb, vec)
                                    if (s > top) top = s
                                }
                                if (top > best) {
                                    best = top
                                    bestName = p.name
                                }
                            }
                            label = if (best > 0.93f) {
                                "$bestName (%.2f)".format(best)
                            } else {
                                "未知 (%.2f)".format(best)
                            }
                        }
                    } else if (isRegistering) {
                        label = "请正对镜头"
                    }

                    DrawInfo(face, label)
                }

                binding.overlay.update(infos, srcW, srcH)
            }
            .addOnFailureListener {
                // 可记录日志
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    /**
     * 由关键点构造一个简易的 16 维特征并做 L2 归一化（演示用；精度有限）。
     * 需要高精度请换为 TFLite/ONNX 的人脸向量模型。
     */
    private fun landmarkEmbedding(face: Face): FloatArray? {
        val box = face.boundingBox
        val fx = box.left.toFloat()
        val fy = box.top.toFloat()
        val fw = box.width().toFloat()
        val fh = box.height().toFloat()
        if (fw <= 0f || fh <= 0f) return null

        fun norm(x: Float, y: Float): Pair<Float, Float> {
            val nx = (x - fx) / max(1f, fw)
            val ny = (y - fy) / max(1f, fh)
            return nx to ny
        }

        val lEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position ?: return null
        val rEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position ?: return null
        val nose = face.getLandmark(FaceLandmark.NOSE_BASE)?.position ?: return null
        val lMouth = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position ?: return null
        val rMouth = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position ?: return null

        val (lex, ley) = norm(lEye.x, lEye.y)
        val (rex, rey) = norm(rEye.x, rEye.y)
        val (nx, ny) = norm(nose.x, nose.y)
        val (lmx, lmy) = norm(lMouth.x, lMouth.y)
        val (rmx, rmy) = norm(rMouth.x, rMouth.y)

        fun dist(ax: Float, ay: Float, bx: Float, by: Float): Float {
            val dx = ax - bx; val dy = ay - by
            return sqrt(dx * dx + dy * dy)
        }

        val eyeDist = dist(lex, ley, rex, rey)
        val eyeNoseL = dist(lex, ley, nx, ny)
        val eyeNoseR = dist(rex, rey, nx, ny)
        val mouthW = dist(lmx, lmy, rmx, rmy)
        val noseMouthL = dist(nx, ny, lmx, lmy)
        val noseMouthR = dist(nx, ny, rmx, rmy)

        val raw = floatArrayOf(
            lex, ley, rex, rey, nx, ny, lmx, lmy, rmx, rmy,
            eyeDist, eyeNoseL, eyeNoseR, mouthW, noseMouthL, noseMouthR
        )

        // L2 归一化
        var norm2 = 0f
        for (v 在 raw) norm2 += v * v
        val s = sqrt(norm2)
        if (s > 0f) {
            for (i 在 raw.indices) raw[i] = raw[i] / s
        }
        return raw
    }
}
