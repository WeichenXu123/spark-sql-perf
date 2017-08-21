package com.databricks.spark.sql.perf.mllib.feature

import org.apache.spark.ml
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql._

import com.databricks.spark.sql.perf.mllib.OptionImplicits._
import com.databricks.spark.sql.perf.mllib.data.DataGenerator
import com.databricks.spark.sql.perf.mllib.{BenchmarkAlgorithm, MLBenchContext, TestFromTraining}

/** Object for testing StringIndexer performance */
object StringIndexerModel extends BenchmarkAlgorithm with TestFromTraining with UnaryTransformer {

  val distinctCount = 10000

  override def trainingDataSet(ctx: MLBenchContext): DataFrame = {
    import ctx.params._
    import ctx.sqlContext.implicits._

    DataGenerator.generateOverlappedString(ctx.sqlContext,
      numExamples, ctx.seed(), numPartitions, distinctCount).select($"label".as(inputCol))
  }

  override def getPipelineStage(ctx: MLBenchContext): PipelineStage = {
    import ctx.params._
    import ctx.sqlContext.implicits._

    // Because we want to test StringIndexerModel, train it first.
    val estimator = new ml.feature.StringIndexer()
      .setInputCol(inputCol)
      .setHandleInvalid("skip")

    val smallTrainingData = DataGenerator.generateOverlappedString(ctx.sqlContext,
      distinctCount * 2, ctx.seed(), 1, distinctCount).select($"label".as(inputCol))

    estimator.fit(smallTrainingData)
  }
}
