package onnx

import org.junit.*
import org.junit.Assert.*

class OnnxTest:

  val path = "res/cat.jpg"
  
  @Test def testSingleExecution() =
    val args = Array(path) 
    val library_time = TypeSafeONNXInference.main(args)
    val raw_time = RawONNXInference.main(args)
    System.out.println("ONNX-scala: "+ library_time)
    System.out.println("ONNX: "+ raw_time)
    assertTrue(library_time > raw_time)
    
  @Test def testBatchExecution() =
    val args = Array(path, "2000") 
    val library_time = TypeSafeONNXInference.main(args)
    val raw_time = RawONNXInference.main(args)
    System.out.println("ONNX-scala: "+ library_time)
    System.out.println("ONNX: "+ raw_time)
    assertTrue(library_time < raw_time)
