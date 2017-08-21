package com.databricks.spark.sql.perf.mllib.feature

import org.apache.spark.ml
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql._
import com.databricks.spark.sql.perf.mllib.OptionImplicits._
import com.databricks.spark.sql.perf.mllib.data.DataGenerator
import com.databricks.spark.sql.perf.mllib.{BenchmarkAlgorithm, MLBenchContext, TestFromTraining}

/** Object for testing VectorAssembler performance */
object VectorAssembler extends BenchmarkAlgorithm with TestFromTraining {

  val numInputCols = 3

  override def trainingDataSet(ctx: MLBenchContext): DataFrame = {
    import ctx.params._
    import ctx.sqlContext.implicits._

    var df: DataFrame = null

    for (i <- 1 to numInputCols) {
      val colName = s"inputCol${i.toString}"
      val newDF = DataGenerator.generateMixedFeatures(
        ctx.sqlContext,
        numExamples,
        ctx.seed(),
        numPartitions,
        Array.fill(numFeatures)(20)
      ).select($"features".as(colName))
      if (df == null) {
        df = newDF
      } else {
        df = df.union(newDF)
      }
    }
    df
  }

  override def getPipelineStage(ctx: MLBenchContext): PipelineStage = {
    import ctx.params._

    val inputCols = (1 to numInputCols)
      .map(i => s"inputCol${i.toString}").toArray

    new ml.feature.VectorAssembler()
      .setInputCols(inputCols)
  }
}