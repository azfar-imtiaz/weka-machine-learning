package com.company;

import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;

public class Attributes_Instances {
    public static void main(String[] args) throws Exception {
        DataSource dataSource = new DataSource("data/diabetes.arff");
        Instances data = dataSource.getDataSet();

        System.out.println("CLASS INFORMATION\n");
        // if the class index is not set, set it to the last attribute in the dataset, which is the class index in our case
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        int numClasses = data.numClasses();
        System.out.println("This dataset has " + numClasses + " unique classes");
        for (int i = 0; i < numClasses; i++) {
            System.out.format("The class at index %d is %s\n", i, data.classAttribute().value(i));
        }
        System.out.println("\n---------------------\n");
        System.out.println("ATTRIBUTE INFORMATION\n");

        int numAttributes = data.numAttributes() - 1;
        for (int i = 0; i < numAttributes; i++) {
            if (data.attribute(i).isNominal()) {
                System.out.format("The %dth attribute is nominal\n", i+1);
                System.out.println("This attribute is: " + data.attribute(i).name());
                int numValues = data.attribute(i).numValues();
                System.out.println("The number of values for this attribute are: " + numValues);
            }

            AttributeStats attStats = data.attributeStats(i);

            if (data.attribute(i).isNumeric()) {
                System.out.format("The %dth attribute is numeric\n", i+1);
                System.out.println("This attribute is: " + data.attribute(i).name());
                Stats numericStats = attStats.numericStats;
                System.out.println("Here are some stats for this numeric attribute: ");
                System.out.println("\tMinimum value: " + numericStats.min);
                System.out.println("\tMaximum value: " + numericStats.max);
                System.out.println("\tAverage value: " + numericStats.mean);
            }

            int distinctCount = attStats.distinctCount;
            int uniqueCount = attStats.uniqueCount;
            System.out.format("This attribute has %d distinct values\n", distinctCount);
            System.out.format("This attribute has %d unique values\n", uniqueCount);

            System.out.println();
        }
        System.out.println("\n---------------------\n");
        System.out.println("INSTANCES INFORMATION\n");

        int numInstances = data.numInstances();
        for (int i = 0; i < numInstances; i++) {
            Instance instance = data.instance(i);

            // This can be extended to check for any missing attribute for any instance - loop till numAttributes
            if (instance.isMissing(0))
                System.out.println("The first attribute is missing for this instance!");

            if (instance.classIsMissing())
                System.out.println("The class is missing for instance " + i);

            // NOTE: This part prints the class value for this instance
            // instance.classValue() returns a double by default
            // int classValue = (int) instance.classValue();
            // System.out.println(instance.classAttribute().value(classValue));
        }
        System.out.println();
    }
}
