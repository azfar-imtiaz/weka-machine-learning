package com.company;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class tfidf {
    public static void main(String[] args) throws Exception {
        DataSource dataSource = new DataSource("data/spamham.arff");
        Instances dataset = dataSource.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataset);
        filter.setIDFTransform(true);
        filter.setLowerCaseTokens(false);
//        filter.setMinTermFreq(2);
        filter.setStemmer(new LovinsStemmer());
        filter.setStopwordsHandler(new Rainbow());

        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(1);
        tokenizer.setNGramMinSize(3);

        filter.setTokenizer(tokenizer);

        // NOTE: This is an alternative way to apply the filter, in case one doesn't want to use FilteredClassifier
//        Instances preprocessedData = Filter.useFilter(dataset, filter);
//        preprocessedData.setClassIndex(preprocessedData.numAttributes() - 1);

        /*
        NOTE: Once a StringToWordVector filter is applied to text data, the resulting Instances data is in a sparse
              vector form. This means that upon exploring or writing to an ARFF file the filtered data, you will note
              that the data will be in the following format:
                {0 1, 5 2, 6 65, 8 90}
                {1 2, 4 1, 13 65}
              This format contains the data in tuples - sort of - where the first index contains the attribute index
              number, and the second index contains the value. Only those attributes are mentioned in this sparse
              format whose value is not 0.
              Additionally, in this sparse format, note that if you have n classes, the data will show class labels
              for n-1 classes, and the class label for the nth class are 'implied'. Therefore, the labels for the last
              class will not be shown in the sparsified data.
         */

        J48 tree = new J48();

        FilteredClassifier clf = new FilteredClassifier();
        clf.setFilter(filter);
        clf.setClassifier(tree);
        clf.buildClassifier(dataset);

        System.out.println(clf.graph());
    }
}
