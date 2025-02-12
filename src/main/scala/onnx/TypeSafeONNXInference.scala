package onnx

import java.nio.file.{Files, Paths}
import scala.io.Source
import org.emergentorder.onnx.backends.ORTModelBackend
import org.emergentorder.onnx.Tensors.Tensor
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.compiletime._
import org.emergentorder.io.kjaer.compiletime._

import cats.effect.unsafe.implicits.global

object TypeSafeONNXInference {
  def main(args: Array[String]): Double = {
    // Load the ONNX model
    val modelPath = "res/squeezenet1.1-7.onnx"
    val squeezenetBytes = Files.readAllBytes(Paths.get("res/squeezenet1.1-7.onnx"))
    val model = new ORTModelBackend(squeezenetBytes)

    // Load and preprocess image
    val dogPath = "res/dog.jpg"
    val catPath = "res/kitten.jpg"

    val imagePath = if (args.nonEmpty) args(0) else catPath
    val image = ImageUtils.loadAndTransform(imagePath)
    // val image = Array.fill(1*3*224*224){24f}

    val iterations = if (args.length > 1) args(1).toInt else 0

    //In NCHW tensor image format
    val shape =                    1     #:     3      #:    224    #: 224     #: SNil
    val tensorShapeDenotation = "Batch" ##: "Channel" ##: "Height" ##: "Width" ##: TSNil

    val tensorDenotation: String & Singleton = "Image"

    val imageTens = Tensor(image,tensorDenotation,tensorShapeDenotation,shape)

    //or as a shorthand if you aren't concerned with enforcing denotations
    val imageTensDefaultDenotations = Tensor(image,shape)

    val t1 = System.nanoTime

    // Perform inference
    val outputs = model.fullModel[Float, 
                               "ImageNetClassification",
                               "Batch" ##: "Class" ##: TSNil,
                               1 #: 1000 #: SNil](Tuple(imageTens))
    for (_ <- 1 to iterations) {
        model.fullModel[Float, 
                        "ImageNetClassification",
                        "Batch" ##: "Class" ##: TSNil,
                        1 #: 1000 #: SNil](Tuple(imageTens))
    }

    val duration = (System.nanoTime - t1) / 1e9d
    println("Executed in: "+duration)

    val outdata = outputs.data.unsafeRunSync()
    // val data: Array[Float] = Array(1.786191E-4, ...)

    //The highest scoring and thus highest probability (predicted) class
    val result = outdata.indices.maxBy(outdata)
    
    // Extract the top 3 predictions
    val topLabels = LabelUtils.postprocessOutput(outdata)
    
    // Print results
    println("Top 3 Predictions:")
    topLabels.foreach { case (label, prob) => println(f"$label: $prob%.4f") }

    duration
  }

  def inferImage(): Seq[String] = {
    
    val imagePath = "res/kitten.jpg"
    val image = ImageUtils.loadAndTransform(imagePath)

    val shape = 1 #: 3 #: 224 #: 224 #: SNil
    val imageTensDefaultDenotations = Tensor(image,shape)

    // Load the ONNX model
    val modelPath = "res/squeezenet1.1-7.onnx"
    val squeezenetBytes = Files.readAllBytes(Paths.get("res/squeezenet1.1-7.onnx"))
    val model = new ORTModelBackend(squeezenetBytes)

        // Perform inference
    val outputs = model.fullModel[Float, 
                               "ImageNetClassification",
                               "Batch" ##: "Class" ##: TSNil,
                               1 #: 1000 #: SNil](Tuple(imageTensDefaultDenotations))

    val outdata = outputs.data.unsafeRunSync()

    val topLabels = LabelUtils.postprocessOutput(outdata)
    
    val result = "Top predictions:" +: topLabels.map { case (label, prob) => f"$label: $prob%.4f" }

    result
  }
}
