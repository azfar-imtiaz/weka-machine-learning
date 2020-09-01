package com.company;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.File;

public class DiscretizeAttributes {
    public static void main(String[] args) throws Exception {
        DataSource dataSource = new DataSource("data/diabetes.arff");
        Instances dataset = dataSource.getDataSet();

        // this means create 4 bins for the first 3 attributes
        // please note that the binning is done for each attribute separately
        String[] options = {"-B", "4", "-R", "1-3"};
        // the -V option inverts the selection, so that the binning is done for all attributes except the first three
        // String[] options = {"-B", "4", "-R", "1-3", "-V"};
        Discretize discretize = new Discretize();
        discretize.setOptions(options);
        discretize.setInputFormat(dataset);
        Instances discretizedData = Filter.useFilter(dataset, discretize);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(discretizedData);
        saver.setFile(new File("data/diabetes-discretized.arff"));
        saver.writeBatch();
    }
}
