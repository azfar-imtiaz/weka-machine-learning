package com.company;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;


public class FilterAttributes {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("data/diabetes.arff");
        Instances dataset = source.getDataSet();

        // this means remove the first and second column. This is NOT 0-indexed!
        String[] options = {"-R", "1, 2"};
        // this means keep only the first and second column
         // String[] options = {"-R", "1, 2", "-V"};

        Remove removal = new Remove();
        removal.setOptions(options);
        removal.setInputFormat(dataset);
        Instances filteredData = Filter.useFilter(dataset, removal);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(filteredData);
        saver.setFile(new File("data/diabetes-filtered.arff"));
        saver.writeBatch();
    }
}
