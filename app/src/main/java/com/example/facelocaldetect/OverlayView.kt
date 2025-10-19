package com.example.facelocaldetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.face.Face

data class DrawInfo(
    val face: Face,
    val label: String? = null
)

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
        color = resources.getColor(R.color.overlayStroke, null)
    }

    private val textPaint = Paint().apply {
        textSize = 42f
        isAntiAlias = true
        color = resources.getColor(R.color.overlayText, null)
    }

    private var infos: List<DrawInfo> = emptyList()
    private var sourceWidth = 1
    private var sourceHeight = 1

    fun update(infos: List<DrawInfo>, srcWidth: Int, srcHeight: Int) {
        this.infos = infos
        this.sourceWidth = srcWidth
        this.sourceHeight = srcHeight
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scaleX = width.toFloat() / sourceWidth.toFloat()
        val scaleY = height.toFloat() / sourceHeight.toFloat()

        for (info in infos) {
            val bounds = info.face.boundingBox
            val rect = RectF(
                bounds.left * scaleX,
                bounds.top * scaleY,
                bounds.right * scaleX,
                bounds.bottom * scaleY
            )
            canvas.drawRect(rect, boxPaint)
            info.label?.let { label ->
                canvas.drawText(label, rect.left, rect.top - 10f, textPaint)
            }
        }
    }
}