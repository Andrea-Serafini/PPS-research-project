package onnx

import org.emergentorder.onnx.Tensors.Tensor
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.backends.ORTOperatorBackendAll
import org.emergentorder.compiletime._
import org.emergentorder.io.kjaer.compiletime._


object TypeSafety {
    val onnxBackend = new ORTOperatorBackendAll()
    val shape =                    1     #:     3      #:    224    #: 224     #: SNil
    val tensorShapeDenotation = "Batch" ##: "Channel" ##: "Height" ##: "Width" ##: TSNil
    val tensorDenotation: String & Singleton = "Image"

    val longTens = Tensor(Array.fill(1*3*224*224){-42l},tensorDenotation,tensorShapeDenotation,shape)
    val floatTens = Tensor(Array.fill(1*3*224*224){42f},tensorDenotation,tensorShapeDenotation,shape)
    onnxBackend.AbsV13("abs", longTens)
    // onnxBackend.SqrtV13("sqrt", longTens)
    onnxBackend.SqrtV13("sqrt", floatTens)

}
