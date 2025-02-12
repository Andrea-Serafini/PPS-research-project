package onnx

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

object ImageUtils {
    def loadAndTransform(imagePath: String, width: Int = 224, height: Int= 224): Array[Float] = {
        val img: BufferedImage = ImageIO.read(new File(imagePath))

        // Resize image
        val resized = img.getScaledInstance(width, height, java.awt.Image.SCALE_SMOOTH)
        val buffered = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
        val g = buffered.getGraphics
        g.drawImage(resized, 0, 0, null)
        g.dispose()

        // Convert to float array and normalize
        val raster = buffered.getRaster
        val pixels = raster.getDataBuffer.asInstanceOf[java.awt.image.DataBufferByte].getData
        val floatPixels = new Array[Float](3 * width * height)

        // SqueezeNet expects input in (C, H, W) order not (HWC)
        for (y <- 0 until height; x <- 0 until width) {
            val i = (y * width + x) * 3 // HWC index
            val r = (pixels(i) & 0xFF) / 255.0f
            val g = (pixels(i + 1) & 0xFF) / 255.0f
            val b = (pixels(i + 2) & 0xFF) / 255.0f

            // Normalize using ImageNet mean & std
            val mean = Array(0.485f, 0.456f, 0.406f)
            val std = Array(0.229f, 0.224f, 0.225f)

            // Store in CHW format
            floatPixels(y * width + x) = (r - mean(0)) / std(0)    // Red
            floatPixels(width * height + y * width + x) = (g - mean(1)) / std(1) // Green
            floatPixels(2 * width * height + y * width + x) = (b - mean(2)) / std(2) // Blue
        }
        floatPixels
    }
}