package com.company;

import weka.core.converters.ArffSaver;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

import java.io.File;

public class AttrSelection {
    public static void main(String[] args) throws Exception {
        DataSource dataSource = new DataSource("data/diabetes.arff");
        Instances dataset = dataSource.getDataSet();

        AttributeSelection filter = new AttributeSelection();
        // there are a bunch of different evaluators that we can choose from
        CfsSubsetEval evaluator = new CfsSubsetEval();
        // there are a bunch of different search techniques we can choose from
        GreedyStepwise search = new GreedyStepwise();
        // I don't know what this does - has something to do with the selected search method
        search.setSearchBackwards(true);

        filter.setEvaluator(evaluator);
        filter.setSearch(search);

        filter.setInputFormat(dataset);

        Instances selectedData = Filter.useFilter(dataset, filter);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(selectedData);
        saver.setFile(new File("data/diabetes-selected.arff"));
        saver.writeBatch();
    }
}
