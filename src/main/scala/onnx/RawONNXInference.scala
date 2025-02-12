package onnx

import ai.onnxruntime._
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._
import ai.onnxruntime.OrtSession.Result

object RawONNXInference {
  def main(args: Array[String]): Double = {
    // Load ONNX model
    val env = OrtEnvironment.getEnvironment
    // val session = new OrtSession(env.createSession("res/squeezenet1.1-7.onnx", new OrtSession.SessionOptions()))
    val session = env.createSession("res/squeezenet1.1-7.onnx", new OrtSession.SessionOptions())


    // Load and preprocess image
    val dogPath = "res/dog.jpg"
    val catPath = "res/kitten.jpg"

    val imagePath = if (args.nonEmpty) args(0) else catPath
    val imageTensor = ImageUtils.loadAndTransform(imagePath)

    val iterations = if (args.length > 1) args(1).toInt else 0

    // Create ONNX input
    val inputName = session.getInputNames.iterator().next()

    // val inputs = Map(inputName -> OnnxTensor.createTensor(env, imageTensor))
    val shape = Array(1L, 3L, 224L, 224L) // Ensure correct input shape
    val inputs = Map(inputName -> OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(imageTensor), shape))

    val t1 = System.nanoTime
    // Run inference
    val results = session.run(inputs.asJava)
    for (_ <- 1 to iterations) {
        session.run(inputs.asJava)
    }

    val duration = (System.nanoTime - t1) / 1e9d
    println("Executed in: "+duration)
    // val outputTensor = results.get(0).getValue.asInstanceOf[Array[Float]]
    val outputTensor = results.get(0).getValue.asInstanceOf[Array[Array[Float]]].head

    // Extract the top 3 predictions
    val topLabels = LabelUtils.postprocessOutput(outputTensor)

    // Print results
    println("Top 3 Predictions:")
    topLabels.foreach { case (label, prob) => println(f"$label: $prob%.4f") }

    // Clean up
    results.close()
    session.close()
    env.close()
    duration
  }

}
