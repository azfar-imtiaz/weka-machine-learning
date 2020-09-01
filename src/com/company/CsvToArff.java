package com.company;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;

public class CsvToArff {
    public static void main(String[] args) throws IOException {
        CSVLoader csvLoader = new CSVLoader();
        csvLoader.setSource(new File("C:/Users/eimtazf/Downloads/SalesJan2009.csv"));
        Instances data = csvLoader.getDataSet();
        System.out.println(data.toSummaryString());

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("data/SalesJan2009.arff"));
        saver.writeBatch();
    }
}
