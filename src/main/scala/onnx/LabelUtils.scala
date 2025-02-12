package onnx

import scala.io.Source

object LabelUtils {
    def postprocessOutput(output: Array[Float]): Seq[(String, Float)] = {
        val labels = Source.fromFile("res/synset.txt").getLines().toList
        val sorted = output.zip(labels).sortBy(-_._1).take(3)
        sorted.map { case (prob, label) => (label, prob) }
    }
}