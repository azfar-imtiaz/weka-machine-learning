package com.company;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;

import java.util.Random;

public class Classification {
    public static void main(String[] args) throws Exception {
        DataSource dataSource = new DataSource("data/diabetes.arff");
        Instances dataset = dataSource.getDataSet();

        double trainTestSplit = 0.8;

        // randomize the dataset
        dataset.randomize(new Random(1));
        // get number of instances for training and testing data as per split - the good old fashioned way
        int numTrainInst = (int) Math.round( (dataset.numInstances() * trainTestSplit) / 100.0 );
        int numTestInst = dataset.numInstances() - numTrainInst;

        // get the training and testing instances
        Instances trainingData = new Instances(dataset, 0, numTrainInst);
        trainingData.setClassIndex(trainingData.numAttributes() - 1);
        Instances testingData = new Instances(dataset, numTrainInst, numTestInst);
        testingData.setClassIndex(testingData.numAttributes() - 1);

//        NaiveBayes clf = new NaiveBayes();
//        nb_clf.buildClassifier(dataset);

        // initialize classifier, train it over training data
        SMO clf = new SMO();
        clf.buildClassifier(trainingData);

//        J48 clf = new J48();
//        String[] options = {"-C", "0.11", "-M", "3"};
//        clf.setOptions(options);
//        clf.buildClassifier(trainingData);
//        // this method is exclusive to decision tree classifiers, of course
//        System.out.println(clf.graph());

        // write model to disk
        SerializationHelper.write("models/svm.model", clf);

        clf = null;
        // the object must be cast to the classifier object type
        clf = (SMO) SerializationHelper.read("models/svm.model");

        // this shows the capabilities and dependencies of the classifier - what kind of situations it can and can't handle
        System.out.println(clf.getCapabilities().toString());
        // this prints classifier specific information - what exactly is printed depends upon the type of classifier
        System.out.println(clf);

        // we provide training data here so the evaluator can get some header and prior class distribution information
        Evaluation eval = new Evaluation(trainingData);
        // we evaluate the classifier on the testing data - duh
        eval.evaluateModel(clf, testingData);

        System.out.println("Classifier performance: ");
        // get classifier evaluation summary
        System.out.println(eval.toSummaryString());
        // print out the confusion matrix
        System.out.println(eval.toMatrixString());
        // print the F1-score - this is done individually, for each class. Same for precision, recall, TPR, NPR etc.
        System.out.println(eval.fMeasure(0));
        System.out.println(eval.fMeasure(1));
    }
}
