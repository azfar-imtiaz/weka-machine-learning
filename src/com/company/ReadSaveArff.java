package com.company;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class ReadSaveArff {

    public static void main(String[] args) throws IOException {
        System.out.println(System.getProperty("user.dir"));
        Instances dataset = new Instances(new BufferedReader(new FileReader("data/diabetes.arff")));
        System.out.println(dataset.toSummaryString());

        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataset);
        saver.setFile(new File("data/diabetes-output.arff"));
        saver.writeBatch();
    }
}
