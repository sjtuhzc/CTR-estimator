/**
  * Created by sjtuhzc on 16-6-1.
  */

/**
  * Created by sjtuhzc on 16-4-26.
  */

package org.apache.spark.mllib.optimization

import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.BLAS.{axpy, dot}
import org.apache.spark.mllib.util.MLUtils
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.mllib.linalg._
import breeze.linalg.{Vector => BV, axpy => brzAxpy, norm => brzNorm}
import java.io._

import breeze.numerics.sqrt


class LRModel(weights:Vector, intercept:Double, rate:Double) extends Serializable{
  val weightMatrix=weights
  def predictPoint(dataMatrix: Vector) = {
    val margin = dot(weightMatrix, dataMatrix) + intercept
    val score = 1.0 / (1.0 + math.exp(-margin))
    score/(score+(1-score)/rate)
  }
}

class LRGradient extends Serializable{
  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }
  def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    val margin = -1.0 * dot(data, weights)
    val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
    axpy(multiplier, data, cumGradient)
    if (label > 0) {
      MLUtils.log1pExp(margin)
    } else {
      MLUtils.log1pExp(margin) - margin
    }

  }
}

class SimpleUpdater extends Serializable{
  def compute(weightsOld: Vector, gradient: Vector, stepSize: Double, iter: Int, regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
  }
}

class SquaredL2Updater extends Serializable{
  def compute(weightsOld: Vector, gradient: Vector, stepSize: Double, iter: Int, regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzWeights :*= (1.0 - thisIterStepSize * regParam)
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    val norm = brzNorm(brzWeights, 2.0)
    (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
  }
}

class LRSGD extends Serializable with Logging{
  private val gradient = new LRGradient()
  private val updater = new SquaredL2Updater()

  def runMiniBatchSGD(data: RDD[(Double, Vector)], stepSize: Double, numIterations: Int, regParam: Double, miniBatchFraction:
  Double, initialWeights: Vector,numPart:Int): (Vector,Int) = {
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size
    var i = 1
    val dataArr=data.randomSplit(Array.fill(numPart)(1.0),11L)
    while (i <= numIterations) {
      for(j<-0 until numPart) {
        val bcWeights = data.context.broadcast(weights)
        val (gradientSum, lossSum, miniBatchSize) = dataArr(j)
          .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
            seqOp = (c, v) => {
              val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
              (c._1, c._2 + l, c._3 + 1)
            },
            combOp = (c1, c2) => {
              (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
            })
        val update = updater.compute(
          weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
          stepSize, i, regParam)
        weights = update._1
      }
      i += 1
    }
    (weights,i)
  }
}


class LRFTRL(numFeatures:Int,iter:Int,a:Double) extends Serializable {
  private val gradient = new LRGradient()
  def compute(data: RDD[(Double, Vector)],ww:Array[Double],n:Array[Double],z:Array[Double]): (Vector,Array[Double],Array[Double]) = {
    val numWeight = numFeatures
    val b = 1.0
    val L1 = 1.0
    val L2 = 1.0
    val minibatch=1.0
    for (it <- 0 until iter) {
      val bcWeights = data.context.broadcast(ww)
      val tmp=data.sample(false, minibatch, 42)
        .treeAggregate((BDV.zeros[Double](numWeight), 0.0, 0L,Vectors.zeros(numWeight)))(
          seqOp = (c, v) => {
            val l = gradient.compute(v._2, v._1, Vectors.dense(bcWeights.value), Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1, v._2)
          },
          combOp = (c1, c2) => {
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3,Vectors.fromBreeze(c1._4.toBreeze+c2._4.toBreeze))
          })
      val g = Vectors.fromBreeze(tmp._1 / minibatch)
      val feature=Vectors.fromBreeze(tmp._4.toBreeze/minibatch)
      feature.foreachActive {
        case (i, v) =>
          var sign = -1.0
          if (z(i) > 0) sign = 1.0
          if (sign * z(i) < L1) ww(i) = 0
          else ww(i) = (sign * L1 - z(i)) / ((b + sqrt(n(i))) / a + L2)
          val sigma = (sqrt(n(i) + g(i) * g(i)) - sqrt(n(i))) / a
          z(i) += g(i) - sigma * ww(i)
          n(i) += g(i) * g(i)
      }
    }
    (Vectors.dense(ww),n,z)
  }
}

class FMModel(val factorMatrix: Matrix,
              val weightVector: Option[Vector],
              numFactors:Int,rate:Double) extends Serializable{
  def predictPoint(testData: Vector): Double = {
    var pred:Double = 0
    testData.foreachActive {
      case (i, v) =>
        pred += weightVector.get(i) * v
    }

    for (f <- 0 until numFactors) {
      var sum = 0.0
      var sumSqr = 0.0
      testData.foreachActive {
        case (i, v) =>
          val d = factorMatrix(f, i) * v
          sum += d
          sumSqr += d * d
      }
      pred += (sum * sum - sumSqr) / 2
    }

    val result=1.0 / (1.0 + Math.exp(-pred))
    result/(result+(1-result)/rate)
  }
}

class FMGradient(val task: Int, val k0: Boolean, val k1: Boolean, val k2: Int,
                 val numFeatures: Int)extends Serializable{

  private def predict(data: Vector, weights: Vector): (Double, Array[Double]) = {

    var pred = weights(weights.size - 1) //globe intercept

    val pos = numFeatures * k2
    data.foreachActive {
      case (i, v) =>
        pred += weights(pos + i) * v
    }

    val sum = Array.fill(k2)(0.0)
    for (f <- 0 until k2) {
      var sumSqr = 0.0
      data.foreachActive {
        case (i, v) =>
          val d = weights(i * k2 + f) * v
          sum(f) += d
          sumSqr += d * d
      }
      pred += (sum(f) * sum(f) - sumSqr) / 2
    }

    (pred, sum)
  }


  private def cumulateGradient(data: Vector, weights: Vector,
                               pred: Double, label: Double,
                               sum: Array[Double], cumGrad: Vector): Unit = {

    val mult = -label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred)))

    cumGrad match {
      case vec: DenseVector =>
        val cumValues = vec.values

        if (k0) {
          cumValues(cumValues.length - 1) += mult
        }

        if (k1) {
          val pos = numFeatures * k2
          data.foreachActive {
            case (i, v) =>
              cumValues(pos + i) += v * mult
          }
        }

        data.foreachActive {
          case (i, v) =>
            val pos = i * k2
            for (f <- 0 until k2) {
              cumValues(pos + f) += (sum(f) * v - weights(pos + f) * v * v) * mult
            }
        }

      case _ =>
        throw new IllegalArgumentException(
          s"cumulateGradient only supports adding to a dense vector but got type ${cumGrad.getClass}.")
    }
  }


  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val cumGradient = Vectors.dense(Array.fill(weights.size)(0.0))
    val loss = compute(data, label, weights, cumGradient)
    (cumGradient, loss)
  }

  def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    require(data.size == numFeatures)
    val (pred, sum) = predict(data, weights)
    cumulateGradient(data, weights, pred, label, sum, cumGradient)
    1 - Math.signum(pred * label)

  }
}

class FMUpdater(val k0: Boolean, val k1: Boolean, val k2: Int,
                val r0: Double, val r1: Double, val r2: Double,
                val numFeatures: Int) extends Serializable{

  def compute(weightsOld: Vector, gradient: Vector,
              stepSize: Double, iter: Int, regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val len = weightsOld.size

    val weightsNew = Array.fill(len)(0.0)
    var regVal = 0.0

    weightsNew(len - 1) = weightsOld(len - 1) - thisIterStepSize * (gradient(len - 1) + r0 * weightsOld(len - 1))
    regVal += r0 * weightsNew(len - 1) * weightsNew(len - 1)
    for (i <- numFeatures * k2 until numFeatures * k2 + numFeatures) {
      weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r1 * weightsOld(i))
      regVal += r1 * weightsNew(i) * weightsNew(i)
    }
    for (i <- 0 until numFeatures * k2) {
      weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r2 * weightsOld(i))
      regVal += r2 * weightsNew(i) * weightsNew(i)
    }

    (Vectors.dense(weightsNew), regVal / 2)
  }
}

class FMFtrl(numFeatures:Int,iter:Int,numfactor:Int) extends Serializable {
  private val gradient = new FMGradient(1,true,true,numfactor,numFeatures)
  def compute(data: RDD[(Double, Vector)],ww:Array[Double],n:Array[Double],z:Array[Double]): (Vector,Array[Double],Array[Double]) = {
    val numWf=numfactor * numFeatures
    val numWeight = numFeatures + numWf + 1
    val a = 0.5
    val b = 1.0
    val L1 = 1.0
    val L2 = 1.0
    val minibatch=1.0
    for (it <- 0 until iter) {
      val bcWeights = data.context.broadcast(ww)
      val tmp=data.sample(false, minibatch, 42)
        .treeAggregate((BDV.zeros[Double](numWeight), 0.0, 0L))(
          seqOp = (c, v) => {
            val l = gradient.compute(v._2, v._1, Vectors.dense(bcWeights.value), Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1)
          },
          combOp = (c1, c2) => {
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })
      val g = Vectors.fromBreeze(tmp._1 / minibatch)
      for(i<-0 until numWeight){
        var sign = -1.0
        if (z(i) > 0) sign = 1.0
        if (sign * z(i) < L1) ww(i) = 0
        else ww(i) = (sign * L1 - z(i)) / ((b + sqrt(n(i))) / a + L2)
        val sigma = (sqrt(n(i) + g(i) * g(i)) - sqrt(n(i))) / a
        z(i) += g(i) - sigma * ww(i)
        n(i) += g(i) * g(i)
      }
    }
    (Vectors.dense(ww),n,z)
  }
}


class FMSGD(numFeatures:Int,numfactor:Int) extends Serializable with Logging{
  private val numfea=numFeatures
  private val gradient = new FMGradient(1,true,true,numfactor,numfea)
  private val updater = new FMUpdater(true,true,numfactor,0, 0.1, 0.1,numfea)

  def runMiniBatchSGD(data: RDD[(Double, Vector)], stepSize: Double, numIterations: Int, regParam: Double, miniBatchFraction: Double, initialWeights: Vector, convergenceTol: Double): Vector = {
    //val numExamples = data.count()
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size
    var i = 1
    while (i <= numIterations) {
      val bcWeights = data.context.broadcast(weights)
      val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
          seqOp = (c, v) => {
            // c: (grad, loss, count), v: (label, features)
            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })
      val update = updater.compute(
        weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
        stepSize, i, regParam)
      weights = update._1
      i += 1
    }
    weights
  }

  def createModel(weights: Vector,rate:Double): FMModel = {

    val values = weights.toArray
    val k2=numfactor

    val v = new DenseMatrix(k2, numFeatures, values.slice(0, numFeatures * k2))

    val w = Some(Vectors.dense(values.slice(numFeatures * k2, numFeatures * k2 + numFeatures)))

    val w0 = values.last

    new FMModel(v, w,k2,rate)
  }
}