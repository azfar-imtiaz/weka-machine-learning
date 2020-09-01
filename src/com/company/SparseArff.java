package com.company;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;

import java.io.File;

public class SparseArff {
    public static void main(String[] args) throws Exception {
        DataSource dataSource = new DataSource("data/before-sparse.arff");
        Instances dataset = dataSource.getDataSet();
        NonSparseToSparse sparseFilter = new NonSparseToSparse();

        sparseFilter.setInputFormat(dataset);
        Instances sparsifiedData = Filter.useFilter(dataset, sparseFilter);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(sparsifiedData);
        saver.setFile(new File("data/after-sparse.arff"));
        saver.writeBatch();
    }
}
