package com.example.facelocaldetect

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
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : ComponentActivity() {

    private lateinit var binding: ActivityMainBinding
    private val executor = Executors.newSingleThreadExecutor()

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
    }

    private val db by lazy { LocalDB(this) }
    private var persons: MutableList<Person> = mutableListOf()

    // Register sampling
    private var isRegistering = false
    private var registerName: String = ""
    private val tempSamples = mutableListOf<FloatArray>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        persons = db.loadAll().toMutableList()

        findViewById<Button>(R.id.btnRegister).setOnClickListener {
            if (isRegistering) {
                Toast.makeText(this, "正在采集，请稍等…", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val input = EditText(this)
            input.hint = "输入姓名"
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

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            persons.clear()
            db.clear()
            Toast.makeText(this, "已清空向量库", Toast.LENGTH_SHORT).show()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
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

            val options = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .enableTracking()
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .build()
            val detector = FaceDetection.getClient(options)

            analysis.setAnalyzer(executor) { imageProxy: ImageProxy ->
                processFrame(imageProxy, detector)
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, selector, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy, detector: com.google.mlkit.vision.face.FaceDetector) {
        val mediaImage = imageProxy.image ?: run { imageProxy.close(); return }
        val rotation = imageProxy.imageInfo.rotationDegrees
        val input = InputImage.fromMediaImage(mediaImage, rotation)

        detector.process(input)
            .addOnSuccessListener { faces: List<Face> ->
                val infos = faces.map { face ->
                    val emb = landmarkEmbedding(face, input.width, input.height)
                    var label: String? = null

                    if (emb != null) {
                        if (isRegistering) {
                            // collect up to 5 samples
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
                            // recognize: cosine to all persons' samples
                            var best = -1f
                            var bestName: String? = null
                            for (p in persons) {
                                // average of top scores among samples
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
                            if (best > 0.93f) { // threshold tuned for this simple embedding
                                label = "$bestName (%.2f)".format(best)
                            } else {
                                label = "未知 (%.2f)".format(best)
                            }
                        }
                    } else {
                        if (isRegistering) {
                            label = "请正对镜头"
                        }
                    }
                    DrawInfo(face, label)
                }
                binding.overlay.update(infos, input.width, input.height)
            }
            .addOnFailureListener {
                // ignore
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    // Build a small, normalized embedding from landmarks (eyes/nose/mouth/ears),
    // using relative positions and distances inside the face bounding box.
    private fun landmarkEmbedding(face: Face, w: Int, h: Int): FloatArray? {
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

        val lEye = face.getLandmark(Face.Landmark.LEFT_EYE)?.position ?: return null
        val rEye = face.getLandmark(Face.Landmark.RIGHT_EYE)?.position ?: return null
        val nose = face.getLandmark(Face.Landmark.NOSE_BASE)?.position ?: return null
        val lMouth = face.getLandmark(Face.Landmark.MOUTH_LEFT)?.position ?: return null
        val rMouth = face.getLandmark(Face.Landmark.MOUTH_RIGHT)?.position ?: return null

        val (lex, ley) = norm(lEye.x, lEye.y)
        val (rex, rey) = norm(rEye.x, rEye.y)
        val (nx, ny) = norm(nose.x, nose.y)
        val (lmx, lmy) = norm(lMouth.x, lMouth.y)
        val (rmx, rmy) = norm(rMouth.x, rMouth.y)

        // features: raw coords + pairwise distances (eyes, eye-nose, mouth width/height, etc.)
        fun dist(ax: Float, ay: Float, bx: Float, by: Float): Float {
            val dx = ax - bx; val dy = ay - by
            return kotlin.math.sqrt(dx*dx + dy*dy)
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
        // L2 normalize
        var norm2 = 0f
        for (v in raw) norm2 += v*v
        val s = kotlin.math.sqrt(norm2)
        if (s > 0f) {
            for (i in raw.indices) raw[i] = raw[i] / s
        }
        return raw
    }
}