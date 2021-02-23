#* test.py
#*
#* ANLY 555 Fall 2020
#* Project Deliverable4
#*
#* Due on: Nov 4, 2020
#* Authors: Leilin Wang, Shengdan Jin, Yifan Zhu, Jieqiao Luo
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than
#* the TAs, professor, textbook and teammates.
#*

########################################################

from toolbox import *


#Test part 1, test for class Dataset
print("\nFollowing is the test for DataSet Object:\n")
print("========================================")


#Test for creating DataSet set object and constuctor
print("\nFollowing is the test for DataSet Constructor:\n")
print('ds = DataSet("Food_Preference.csv")')
ds = DataSet("Food_Preference.csv")
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function explore in DataSet
#newds=ds.loadDataset()

#Test for function clean in DataSet
print("\nFollowing is the test for DataSet Clean Function:\n")
print('ds.clean()')
ds.clean()
print("\nClean run succesfully\n")
print("________________________________________")

#Test for function explore in DataSet
print("\nFollowing is the test for DataSet Explore Function:\n")
print('ds.explore()')
ds.explore()
print("\nExplore run succesfully\n")
print("________________________________________")

#Test for function summary in DataSet
print("\nFollowing is the test for DataSet Summary Function:\n")
print('ds.summary()')
ds.summary()
print("\nSummary run succesfully\n")
print("________________________________________")

#Test for function head
print("\nFollowing is the test for DataSet Head Function:\n")
print('ds.head()')
ds.head()
print("\nHead run succesfully\n")
print("________________________________________")

#Test for function getColumnName in DataSet
print("\nFollowing is the test for DataSet getColumnName Function:\n")
print('ds.getColumnName()')
print(ds.getColumnName())
print("\ngetColumneName run succesfully\n")
print("________________________________________")

#Test for function setColumnName in DataSet
print("\nFollowing is the test for DataSet setColumnName Function:\n")
print('ds.setColumnName("Age","Ages")')
ds.setColumnName("Age","Ages")
print("\nsetColumnName run succesfully\n")
print("________________________________________")


#Test for function printDS
print("\nFollowing is the test for DataSet printDS Function:\n")
print('ds.printDS()')
ds.printDS()
print("\nprintDS run succesfully\n")
print("________________________________________")

#Test for function getUniqueCount
print("\nFollowing is the test for DataSet getUniqueCount Function, use Gender as Example:\n")
print('ds.getUniqueCount("Gender")')
print(ds.getUniqueCount("Gender"))
print("\ngetUniqueCount run succesfully\n")
print("________________________________________")


#Test for function getColumn
print("\nFollowing is the test for DataSet getUniqueCount Function, use Food as Example:\n")
print('ds.getColumn("Food")[:5,]')
print(ds.getColumn("Food")[:5,])
print("\ngetUniqueCount run succesfully\n")
print("________________________________________")

#Test for function removeColumn
print("\nFollowing is the test for DataSet removeColumn Function, use Food as Example:\n")
print("Before remove:")
print(ds.getColumnName())
print('After use function ds.removeColumn("Food"), the new dataset column:')
print(ds.removeColumn("Food").getColumnName())
print("\nremoveColumn run succesfully\n")
print("________________________________________")

#Test for function savetoCSV
print("\nFollowing is the test for DataSet savetoCSV Function to save new test.csv:\n")
print('ds.savetoCSV("test.csv")')
ds.savetoCSV("test.csv")
print("\nsavetoCSV run succesfully\n")
print("________________________________________")
print("\nThat's the end of the test of DataSet Object\n")
print("========================================")

########################################################

#Test part 2, test for class TimeSeriesDataset
print("========================================")
print("========================================")
print("\nFollowing is the test for TimeSeriesDataset Object:\n")
print("========================================")

#Test for creating TimeSeriesDataSet set object and constuctor
print("\nFollowing is the test for TimeSeriesDataSet Constructor:\n")
print('tsds = TimeSeriesDataSet("AABA_2006-01-01_to_2018-01-01.csv")')
tsds = TimeSeriesDataSet("AABA_2006-01-01_to_2018-01-01.csv")
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function sort in TimeSeriesDataSet
print("\nFollowing is the test for TimeSeriesDataSet sort Function\n")
print("tsds.sort()")
tsds.sort()
tsds.printDS()
print("(since the dataset is orginally sorted, the result is the same)")
print("\nsort run succesfully\n")
print("________________________________________")

#Test for function setPeriod in TimeSeriesDataSet
print("\nFollowing is the test for TimeSeriesDataSet setPeriod Function:\n")
print('tsds.setPeriod("2006-01-03","2008-01-13")')
tsds.setPeriod('2006-01-03','2008-01-16')
print("\nsetPeriod run succesfully\n")
print("________________________________________")

#Test for function PeriodData in TimeSeriesDataSet
print("\nFollowing is the test for TimeSeriesDataSet PeriodData Function:\n")
print("The original dataset before test")
print(tsds.printDS())
print("The period dataset after period set with following function:")
print("tsds.PeriodData()")
print(tsds.PeriodData())
print("\nPeriodData run succesfully\n")
print("________________________________________")

#Test for function changeType in TimeSeriesDataSet
print("\nFollowing is the test for TimeSeriesDataSet changeType Function:\n")
print('tsds.changeType(0,"M")')
tsds.changeType(0,'M')
print("\nchangeType run succesfully\n")
print("________________________________________")

#Test for function summary in TimeSeriesDataSet
print("\nFollowing is the test for TimeSeriesDataSet summary Function:\n")
print('tsds.summary()')
tsds.summary()
print("\nsummary run succesfully\n")
print("________________________________________")

#Test for function clean in TimeSeriesDataSet
print("\nFollowing is the test for TimeSeriesDataSet clean Function, use 3 as filter example:\n")
print("shape before cleaning")
tsds.summary()
print('shape after cleaning with function tsds.clean(3)')
tsds.clean(3)
tsds.summary()
print("\nclean run succesfully\n")
print("________________________________________")

#Test for function explore in TimeSeriesDataSet
print("\nFollowing is the test for TimeSeriesDataSet explore Function, use column open as example:\n")
print("tsds.explore('Open')")
tsds.explore("Open")
print("\nExplore run successfully. Two plotly was opened, first is timeline graph for open and filted open, second is the stcok change in the period\n")
print("________________________________________")
print("\nThat's the end of the test of TimeSeriesDataSet Object\n")
print("========================================")
      
########################################################


#Test part 3, test for class TextDataSet
print("========================================")
print("========================================")
print("\nFollowing is the test for TextDataSet Object:\n")
print("========================================")

#Test for creating TextDataSet set object and constuctor
print("\nFollowing is the test for TextDataSet Constructor:\n")
print('tds = TextDataSet("yelp.csv")')
tds = TextDataSet("yelp.csv")
print("\nConstructor run succesfully\n")
print("________________________________________")


#Test for function clean in TextDataSet
print("\nFollowing is the test for TextDataSet clean Function:\n")
print('tds.clean()')
tds.clean()
print("\nClean run succesfully\n")
print("________________________________________")

#Test for function explore in TextDataSet
print("\nFollowing is the test for TextDataSet explore Function:\n")
print('tds.explore()')
tds.explore()
print("\nExplore run successfully. Two plot was generated, first is wordcloud for text data, second is result bar graph\n")
print("________________________________________")


#Test for function summary in TextDataSet
print("\nFollowing is the test for TextDataSet summary Function:\n")
print('tds.summary()')
tds.summary()
print("\nSummary run succesfully\n")
print("________________________________________")

#Test for function getText in TextDataSet
print("\nFollowing is the test for TextDataSet getText Function, output only first 4 rows:\n")
print('tds.getText()[:3]')
print(tds.getText()[:3])
print("\ngetText run succesfully\n")
print("________________________________________")

#Test for function remove_number in TextDataSet
print("\nFollowing is the test for TextDataSet remove_number Function:\n")
print("\n*****The first 4 rows before remove_number using tds.getText()[:3]*****\n")
print(tds.getText()[:3])
print('\n*****The first 4 rows after remove_number using tds.remove_number()[:3]*****\n')
print(tds.remove_number()[:3])
print("\nremove_number run succesfully\n")
print("________________________________________")


#Test for function remove_punctuation in TextDataSet
print("\nFollowing is the test for TextDataSet remove_punctuation Function:\n")
print("\n*****The first 4 rows before remove_punctuation using tds.getText()[:3]*****\n")
print(tds.getText()[:3])
print('\n*****The first 4 rows after remove_punctuation using tds.remove_punctuation()[:3]*****\n')
print(tds.remove_punctuation()[:3])
print("\nremove_punctuation run succesfully\n")
print("________________________________________")


#Test for function stemming in TextDataSet
print("\nFollowing is the test for TextDataSet stemming Function:\n")
print("\n*****The first 4 rows before stemming using tds.getText()[:3]*****\n")
print(tds.getText()[:3])
print('\n*****The first 4 rows after stemming using tds.stemming()[:3]*****\n')
print(tds.stemming()[:3])
print("\nstemming run succesfully\n")
print("________________________________________")

#Test for function lemmatization in TextDataSet
print("\nFollowing is the test for TextDataSet lemmatization Function:\n")
print("\n*****The first 4 rows before lemmatization using tds.getText()[:3]*****\n")
print(tds.getText()[:3])
print('\n*****The first 4 rows after lemmatization using tds.lemmatization()[:3]*****\n')
print(tds.lemmatization()[:3])
print("\nlemmatization run succesfully\n")
print("________________________________________")

#Test for function tokenize in TextDataSet
print("\nFollowing is the test for TextDataSet tokenize Function:\n")
print("\n*****The first 4 rows before tokenize using tds.getText()[:3]*****\n")
print(tds.getText()[:3])
print('\n*****The first 4 rows of tokenize list after tokenize using tds.tokenize()[:3]*****\n')
print(tds.tokenize()[:3])
print("\ntokenize run succesfully\n")
print("________________________________________")
print("\nThat's the end of the test of TextDataSet Object\n")
print("========================================")

########################################################

#Test part 4, test for class QuantDataSet
print("========================================")
print("========================================")
print("\nFollowing is the test for QuantDataSet Object:\n")
print("========================================")


#Test for creating QuantDataSet set object and constuctor
print("\nFollowing is the test for QuantDataSet Constructor:\n")
print('qnds = QuantDataSet("Sales_Transactions_Dataset_Weekly.csv")')
qnds = QuantDataSet("Sales_Transactions_Dataset_Weekly.csv")
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function clean in QuantDataSet
print("\nFollowing is the test for QuantDataSet clean Function:\n")
print('qnds.clean()')
qnds.clean()
print("(since the dataset is orginally cleaned, the result is the same)")
print("\nclean run succesfully\n")
print("________________________________________")

#Test for function explore in QuantDataSet
print("\nFollowing is the test for QuantDataSet explore Function, use product 3 for fig1 and W1 for fig2 as an example:\n")
print('qnds.explore(3,"W1")')
qnds.explore(3,"W1")
print("\nExplore run successfully. Two plot was generated, first is line graph for specific product, second is bar graph for products in a week\n")
print("________________________________________")

#Test for function summary in QuantDataSet
print("\nFollowing is the test for QuantDataSet summary Function, use column 2, 4, 56, 58 for week0 and week2 summary:\n")
print('qnds.summary([1,3,55,57])')
qnds.summary([1,3,55,57])
print("\nsummary run succesfully\n")
print("________________________________________")

#Test for function getColumnName in QuantDataSet
print("\nFollowing is the test for QuantDataSet getColumnName Function:\n")
print('qnds.getColumnName()')
print(qnds.getColumnName())
print("\ngetColumneName run succesfully\n")
print("________________________________________")

#Test for function setColumnName in QuantDataSet
print("\nFollowing is the test for QuantDataSet setColumnName Function:\n")
print('qnds.setColumnName("MAX","maximum")')
print(qnds.setColumnName("MAX","maximum"))
print("\nsetColumnName run succesfully\n")
print("________________________________________")

#Test for function sort in QuantDataSet
print("\nFollowing is the test for QuantDataSet sort Function:\n")
print('qnds.sort("W1")')
print(qnds.sort("W1"))
print("\nsort run succesfully\n")
print("________________________________________")
print("\nThat's the end of the test of QuantDataSet Object\n")
print("========================================")
      

########################################################


#Test part 5, test for class QualDataSet
print("========================================")
print("========================================")
print("\nFollowing is the test for QualDataSet Object:\n")
print("========================================")


#Test for creating QualDataSet set object and constuctor
print("\nFollowing is the test for QualDataSet Constructor:\n")
print('qlds = QualDataSet("Food_Preference.csv")')
qlds = QualDataSet("Food_Preference.csv")
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function clean in QualDataSet
print("\nFollowing is the test for QualDataSet clean Function:\n")
print('qnds.clean()')
qlds.clean()
print("\nclean run succesfully\n")
print("________________________________________")

#Test for function explore in QualDataSet
print("\nFollowing is the test for QualDataSet explore Function, use nationality for pie chart and age and gender for bar chart as an example:\n")
print('qlds.explore("Nationality","Age","Gender")')
qlds.explore("Nationality","Age","Gender")
print("\nexplore run succesfully\n")
print("________________________________________")

#Test for function summary in QualDataSet
print("\nFollowing is the test for QualDataSet summary Function, only column of contains unique value of more than 2 counts will be output:\n")
print('qlds.summary()')
qlds.summary()
print("\nsummary run succesfully\n")
print("________________________________________")

#Test for function getColumnName in QualDataSet
print("\nFollowing is the test for QualDataSet getColumnName Function:\n")
print('qlds.getColumnName()')
print(qlds.getColumnName())
print("\ngetColumneName run succesfully\n")
print("________________________________________")

#Test for function setColumnName in QualDataSet
print("\nFollowing is the test for QualDataSet setColumnName Function:\n")
print('qlds.setColumnName("Gender","G")')
print(qlds.setColumnName("Gender","G"))
print("\nsetColumnName run succesfully\n")
print("________________________________________")

#Test for function removeColumn in QualDataSet
print("\nFollowing is the test for QualDataSet removeColumn Function, use Timestamp as Example:\n")
print("Before remove:")
print(qlds.getColumnName())
print('After use function ds.removeColumn("Timestamp"), the new QualDataSet column:')
print(qlds.removeColumn("Timestamp").getColumnName())
print("\nremoveColumn run succesfully\n")
print("________________________________________")
print("\nThat's the end of the test of QuantDataSet Object\n")
print("========================================")

########################################################


#Test part 6,test for class TransactionDataSet
print("========================================")
print("========================================")
print("\nFollowing is the test for TransactionDataSet Object:\n")
print("========================================")

#Test for creating TransactionDataSet set object and constuctor
print("\nFollowing is the test for TransactionDataSet Constructor:\n")
print('tsacds = TransactionDataSet("GroceryStoreDataSet.csv")') 
tsacds = TransactionDataSet("GroceryStoreDataSet.csv")
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function clean in TransactionDataSet
print("\nFollowing is the test for TransactionDataSet clean Function:\n")
print('tsacds.clean()')
tsacds.clean()
print("\nclean run succesfully\n")
print("________________________________________")

#Test for function FrequentItem in TransactionDataSet
print("\nFollowing is the test for TransactionDataSet FrequentItem Function:\n")
print('tsacds.FrequentItem(0.15)')
tsacds.FrequentItem(0.15)
print("\nFrequentItem run succesfully\n")
print("________________________________________")

#Test for function explore in TransactionDataSet
print("\nFollowing is the test for TransactionDataSet explore Function:\n")
print("tsacds.explore()")
tsacds.explore()
print("\nExplore run successfully. Top 10 rules with Support, Confidence and Lift are printed \n")
print("________________________________________")

#Test for function summary in TransactionDataSet
print("\nFollowing is the test for TransactionDataSet summary Function:\n")
print('tsacds.summary()')
tsacds.summary()
print("\nsummary run succesfully\n")
print("________________________________________")
print("\nThat's the end of the test of TransactionDataSet Object\n")
print("========================================")

########################################################


#Test part 7, test for class ClassifierAlgorithm
print("========================================")
print("========================================")
print("\nFollowing is the test for ClassifierAlgorithm Object:\n")
print("========================================")


#Test for creating ClassifierAlgorithm set object and constuctor
print("\nFollowing is the test for ClassifierAlgorithm Constructor:\n")
print('ca = ClassifierAlgorithm()')
ca = ClassifierAlgorithm()
print("\nConstructor run succesfully\n")
print("________________________________________")


#Test for function train in ClassifierAlgorithm
print("\nFollowing is the test for ClassifierAlgorithm train Function:\n")
print('First create dataset object with function qlds = QualDataSet("knndata.csv")')
qlds = QualDataSet("knndata.csv")
print('The dataset is created, then use following train function to train dataset and save dataset, train size is 400')
print('ca.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])')
ca.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])
print("\nTrain run succesfully\n")
print("________________________________________")

#Test for function test in ClassifierAlgorithm
print("\nFollowing is the test for ClassifierAlgorithm test Function:\n")
print('ca.test(qlds.getds()[401:,2:])')
print(ca.test(qlds.getds()[401:,2:]))
print("\nTest run succesfully, test data is saved\n")
print("________________________________________")

#Test for function setRatio in ClassifierAlgorithm
print("\nFollowing is the test for ClassifierAlgorithm setRatio Function, set the train ratio as 0.8 for an example:\n")
print('ca.setRatio(0.8)')
ca.setRatio(0.8)
print("\nsetRatio run succesfully\n")
print("________________________________________")

#Test for function splitTrainTest in ClassifierAlgorithm
print("\nFollowing is the test for ClassifierAlgorithm splitTrainTest Function, use the preset the train ratio of 0.8,")
print ("following function output the shape of the auto split test set size:\n")
print('ca.splitTrainTest(qlds.removeColumn("id").removeColumn("diagnosis").getds(),qlds.getColumn("diagnosis")).shape')
print(ca.splitTrainTest(qlds.removeColumn("id").removeColumn("diagnosis").getds(),qlds.getColumn("diagnosis")).shape)
print("\nsplitTrainTest run succesfully\n")
print("________________________________________")
print("\nThat's the end of the test of ClassifierAlgorithm Object\n")
print("========================================")

########################################################

#Test part 8, test for class simplekNNClassifier
print("========================================")
print("========================================")
print("\nFollowing is the test for simplekNNClassifier Object:\n")
print("========================================")


#Test for creating simplekNNClassifier set object and constuctor
print("\nFollowing is the test for simplekNNClassifier Constructor:\n")
print('skc = simplekNNClassifier()')
skc = simplekNNClassifier()
print("\nConstructor run succesfully\n")
print("________________________________________")


#Test for function train in simplekNNClassifier
print("\nFollowing is the test for simplekNNClassifier train Function:\n")
print('First create dataset object with function qlds = QualDataSet("knndata.csv")')
qlds = QualDataSet("knndata.csv")
print('The dataset is created, then use following train function to train dataset and save dataset, train size is 400')
print('skc.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])')
skc.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])
print("\nTrain run succesfully\n")
print("________________________________________")

#Test for function setK in simplekNNClassifier
print("\nFollowing is the test for simplekNNClassifier setK Function, set k as 7 for example:\n")
print('skc.setK(7)')
skc.setK(7)
print("\nsetK run succesfully\n")
print("________________________________________")

#Test for function test in simplekNNClassifier
print("\nFollowing is the test for simplekNNClassifier test Function, output 20 predictions for example:\n")
print('skc.test(qlds.getds()[401:,2:])[0][:19]')
print(skc.test(qlds.getds()[401:,2:])[0][:19])
print("\nTest run succesfully, test data is saved\n")
print("________________________________________")

#Test for function splitTrainTest in simplekNNClassifier
print("\nFollowing is the test for simplekNNClassifier splitTrainTest Function, use the default the train ratio of 0.7,")
print ("following function output the shape of the auto split test set size:\n")
print('skc.test(skc.splitTrainTest(qlds.removeColumn("id").removeColumn("diagnosis").getds(),qlds.getColumn("diagnosis")))[0][:19]')
print(skc.test(skc.splitTrainTest(qlds.removeColumn("id").removeColumn("diagnosis").getds(),qlds.getColumn("diagnosis")))[0][:19])
print("\nsplitTrainTest run succesfully\n")
print("________________________________________")

#Test for toString function in simplekNNClassifier
print("\nFollowing is the test for simplekNNClassifier toString Function:\n")
print('skc.toString()')
print(skc.toString())
print("\ntoString run successfully.\n")
print("________________________________________")

#Test for drawPrediction function in simplekNNClassifier
print("\nFollowing is the test for simplekNNClassifier drawPrediction Function:\n")
print('skc.drawPrediction()')
skc.drawPrediction()
print("\ndrawPrediction run successfully. One bar graph of plot is opened\n")
print("________________________________________")
print("\nThat's the end of the test of simplekNNClassifier Object\n")
print("========================================")

########################################################

#Test part 9, test for class DecisionTree
print("========================================")
print("========================================")
print("\nFollowing is the test for DecisionTree Object:\n")
print("========================================")


#Test for creating DecisionTree set object and constuctor
print("\nFollowing is the test for DecisionTree Constructor:\n")
print('dt = DecisionTree()')
dt = DecisionTree()
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function train in DecisionTree
print("\nFollowing is the test for DecisionTree train Function for continuous Data:\n")
print('First create dataset object with function qlds = QualDataSet("knndata.csv")')
qlds = QualDataSet("knndata.csv")
print('The dataset is created, then use following train function to train dataset and save dataset, train size is 400')
print('dt.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])')
dt.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])
print("\nTrain run succesfully\n")
print("________________________________________")


#Test for function test in DecisionTree
print("\nFollowing is the test for DecisionTree test Function for continuous Data, output 20 predictions for example:\n")
print('dt.test(qlds.getds()[401:,2:])[0][:19]')
print(dt.test(qlds.getds()[401:,2:])[0][:19])
print("\nTest run succesfully, test data is saved\n")
print("________________________________________")


#Test for toString function in DecisionTree
print("\nFollowing is the test for DecisionTree toString Function:\n")
print('dt.toString()')
print(dt.toString())
print("\ntoString run successfully.\n")
print("________________________________________")

#Test for drawPrediction function in DecisionTree
print("\nFollowing is the test for DecisionTree drawPrediction Function:\n")
print('dt.drawPrediction()')
dt.drawPrediction()
print("\ndrawPrediction run successfully. One bar graph of plot is opened\n")
print("\nTest for continuous decision tree run successfully\n")
print("________________________________________")


#Test for function train in DecisionTree
print("\nFollowing is the test for DecisionTree train Function for discrete Data:\n")
print('First create dataset object with function qlds = QualDataSet("Food_Preference.csv"), and clean with qlds.clean()')
qlds = QualDataSet("Food_Preference.csv")
qlds.clean()
print('The dataset is created, then use following train function to train dataset and save dataset, train size is 210')
print('dt.train(qlds.getds()[1:210,5:],qlds.getds()[1:210,2])')
dt.train(qlds.getds()[1:210,5:],qlds.getds()[1:210,2])
print("\nTrain run succesfully\n")
print("________________________________________")

#Test for function setType in DecisionTree
print("\nFollowing is the test for DecisionTree setType Function, set Type as 'discrete' for discrete data:\n")
print('dt.setType("discrete")')
dt.setType("discrete")
print("\nsetType run succesfully\n")
print("________________________________________")

#Test for function test in DecisionTree
print("\nFollowing is the test for DecisionTree test Function for continuous Data, output 20 predictions for example:\n")
print('dt.test(qlds.getds()[210:,5:])[0][:19]')
print(dt.test(qlds.getds()[210:,5:])[0][:19])
print("\nTest run succesfully, test data is saved\n")
print("________________________________________")


#Test for toString function in DecisionTree
print("\nFollowing is the test for DecisionTree toString Function:\n")
print('dt.toString()')
print(dt.toString())
print("\ntoString run successfully.\n")
print("________________________________________")

#Test for drawPrediction function in DecisionTree
print("\nFollowing is the test for DecisionTree drawPrediction Function:\n")
print('dt.drawPrediction()')
dt.drawPrediction()
print("\ndrawPrediction run successfully. One bar graph of plot is opened\n")
print("\nTest for discrete decision tree run successfully\n")
print("________________________________________")
print("\nThat's the end of the test of DecisionTree Object\n")
print("========================================")

########################################################

#Test part 10, test for class Experiment
print("========================================")
print("========================================")
print("\nFollowing is the test for Experiment Object:\n")
print("========================================")

from toolbox import *

#Test for creating Experiment set object and constuctor
print("\nFollowing is the test for Experiment Constructor:\n")
print('First create dataset object with function qlds = QualDataSet("knndata.csv")')
qlds = QualDataSet("knndata.csv")
print('Then create two classifier object, skc for knn and dt for decision tree, with following function')
print('skc = simplekNNClassifier()')
skc = simplekNNClassifier()
print('dt = DecisionTree()')
dt = DecisionTree()
print('Then set k value as 5 for simplekNNClassifier object skc and set type as continuous for DecisionTree object dt with following function')
print('skc.setK(5)')
skc.setK(5)
print('dt.setType("continuous")')
dt.setType("continuous")
print('Finally, create experiment object with following function and parameter')
print('exp = Experiment(qlds.removeColumn("id").removeColumn("diagnosis").getds(),qlds.getColumn("diagnosis"),[skc,dt])')
exp = Experiment(qlds.removeColumn("id").removeColumn("diagnosis").getds(),qlds.getColumn("diagnosis"),[skc,dt])
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function runCrossVal in Experiment
print("\nFollowing is the test for Experiment runCrossVal Function, use 5 cross validation folder:\n")
print('exp.runCrossVal(5)')
exp.runCrossVal(5)
print("\nrunCrossVal run successfully, prediction label of each classifier is saved.\n")
print("________________________________________")

#Test for function score in Experiment
print("\nFollowing is the test for Experiment score Function:\n")
print('exp.score()')
exp.score()
print("\nscore run successfully\n")
print("________________________________________")

#Test for function confusionMatrix in Experiment
print("\nFollowing is the test for Experiment confusionMatrix Function:\n")
print('exp.confusionMatrix()')
exp.confusionMatrix()
print("\nconfusionMatrix run successfully\n")
print("________________________________________")


#Test for function drawConfusionMatrix in Experiment
print("\nFollowing is the test for Experiment drawConfusionMatrix Function, for example, draw the confusion matrix of first classifier with index 0:\n")
print('exp.drawConfusionMatrix(0)')
exp.drawConfusionMatrix(0)
print("\ndrawConfusionMatrix run successfully, one plotly of confusion matrix heat map is opened\n")
print("________________________________________")
print("\nThat's the end of the test of Experiment Object\n")
print("========================================")


#Test for function ROC in Experiment
print("\nFollowing is the test for Experiment ROC Function:\n")
print('exp.ROC_Curve()')
exp.ROC_Curve()
print("\ndrawConfusionMatrix run successfully, one ROC plot is generated\n")
print("________________________________________")
print("\nThat's the end of the test of Experiment Object\n")
print("========================================")

#Test for function ROC for multiclass in Experiment
print("\nFollowing is the test for Experiment Function with multilabel:\n")
print('Use function iris = QualDataSet("Iris.csv") to input iris dataset')
iris = QualDataSet("Iris.csv")
print('Create experiment object with following function and parameter')
print('exp2 = Experiment(iris.removeColumn("Id").removeColumn("Species").getds(),iris.getColumn("Species"),[skc,dt])')
exp2 = Experiment(iris.removeColumn("Id").removeColumn("Species").getds(),iris.getColumn("Species"),[skc,dt])
print('Run cross validation with exp2.runCrossVal(10)')
exp2.runCrossVal(10)
print('Output confusion matrix with exp2.confusionMatrix()')
exp2.confusionMatrix()
print('Draw ROC curve with exp2.ROC_Curve()')
exp2.ROC_Curve()
print("\ndrawConfusionMatrix run successfully, one ROC plot is generated\n")
print("\nTest for mulitlabel experiment object run successfully\n")
print("________________________________________")
print("\nThat's the end of the test of Experiment Object\n")
print("========================================")


########################################################


#Test part 11, test for class HeterogenousDataSets
print("========================================")
print("========================================")
print("\nFollowing is the test for HeterogenousDataSets Object:\n")
print("========================================")

#Test for creating HeterogenousDataSets set object and constuctor
print("\nFollowing is the test for HeterogenousDataSets Constructor:\n")
print('htrds = HeterogenousDataSets(["yelp.csv","GroceryStoreDataSet.csv"])') 
htrds = HeterogenousDataSets(["yelp.csv","GroceryStoreDataSet.csv"])
print("\nConstructor run succesfully\n")
print("________________________________________")


#Test for function loadDataset in HeterogenousDataSets
print("\nFollowing is the test for HeterogenousDataSets loadDataset Function:\n")
print('htrds.loadDataset(["TextDataSet","TransactionDataSet"])')
htrds.loadDataset(["TextDataSet","TransactionDataSet"])
print("\nloadDataset run succesfully\n")
print("________________________________________")


#Test for function clean in HeterogenousDataSets
print("\nFollowing is the test for HeterogenousDataSets clean Function:\n")
print('htrds.clean()')
htrds.clean()
print("\nclean run succesfully\n")
print("________________________________________")


#Test for function explore in HeterogenousDataSets
print("\nFollowing is the test for HeterogenousDataSets explore Function:\n")
print("htrds.explore()")
htrds.explore()
print("\nExplore run successfully\n")
print("________________________________________")


#Test for function select in HeterogenousDataSets
print("\nFollowing is the test for HeterogenousDataSets select Function:\n")
print("htrds.select(0)")
htrds.select(0)
print("\nSelect run successfully\n")
print("________________________________________")



#Test for function summary in HeterogenousDataSets
print("\nFollowing is the test for HeterogenousDataSets summary Function:\n")
print('htrds.summary()')
htrds.summary()
print("\nsummary run succesfully\n")
print("________________________________________")
print("\nThat's the end of the test of HeterogenousDataSets Object\n")
print("========================================")

########################################################

#Test part 12,test for creating lshkNNClassifier set object and constuctor
print("\nFollowing is the test for lshkNNClassifier Constructor:\n")
print('lshk = lshkNNClassifier(10,20)')
lshk = lshkNNClassifier(10,20)
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function train in lshkNNClassifier
print("\nFollowing is the test for lshkNNClassifier train Function for continuous Data:\n")
print('First create dataset object with function qlds = QualDataSet("knndata.csv")')
qlds = QualDataSet("knndata.csv")
print('The dataset is created, then use following train function to train dataset and save dataset, train size is 400')
print('lshk.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])')
lshk.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])
print("\nTrain run succesfully\n")
print("________________________________________")


#Test for function test in lshkNNClassifier
print("\nFollowing is the test for lshkNNClassifier test Function for continuous Data, output 20 predictions for example:\n")
print('lshk.test(qlds.getds()[401:,2:])[0][:19]')
print(lshk.test(qlds.getds()[401:,2:])[0][:19])
print("\nTest run succesfully, test data is saved\n")
print("________________________________________")

#Test for toString function in lshkNNClassifier
print("\nFollowing is the test for lshkNNClassifier toString Function:\n")
print('lshk.toString()')
print(lshk.toString())
print("\ntoString run successfully.\n")
print("________________________________________")

#Test for drawPrediction function in lshkNNClassifier
print("\nFollowing is the test for lshkNNClassifier drawPrediction Function:\n")
print('lshk.drawPrediction()')
lshk.drawPrediction()
print("\ndrawPrediction run successfully. One bar graph of plot is opened\n")
print("\nTest for lshkNNClassifier run successfully\n")
print("________________________________________")

#Test for drawPrediction function in lshkNNClassifier
print("\nFollowing is the test for accuracy compare between lshkNNClassifier and simpleKNNClassifier Using Experiment:\n")
print('First create both object skc = simplekNNClassifier()')
skc = simplekNNClassifier()
print('Then set k value as 10 for simplekNNClassifier object skc')
print('skc.setK(10)')
skc.setK(10)
print('Create experiment object with following function and parameter')
print('exp3 = Experiment(qlds.removeColumn("id").removeColumn("diagnosis").getds(),qlds.getColumn("diagnosis"),[skc,lshk])')
exp3 = Experiment(qlds.removeColumn("id").removeColumn("diagnosis").getds(),qlds.getColumn("diagnosis"),[skc,lshk])
print('Run cross validation with exp3.runCrossVal(10)')
exp3.runCrossVal(10)
print('Output accuracy with exp3.score()')
exp3.score()
print("________________________________________")
print("\nThat's the end of the test of lshkNNClassifier Object\n")
print("========================================")
########################################################

#Test part 13,test for creating kdTreeKNNClassifier set object and constuctor
print("\nFollowing is the test for kdTreeKNNClassifier Constructor:\n")
print('knnkdt = kdTreeKNNClassifier()')
knnkdt = kdTreeKNNClassifier()
print("\nConstructor run succesfully\n")
print("________________________________________")

#Test for function train in kdTreeKNNClassifier
print("\nFollowing is the test for kdTreeKNNClassifier train Function for continuous Data:\n")
print('First create dataset object with function qlds = QualDataSet("knndata.csv")')
qlds = QualDataSet("knndata.csv")
print('The dataset is created, then use following train function to train dataset and save dataset, train size is 400')
print('knnkdt.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])')
knnkdt.train(qlds.getds()[1:400,2:],qlds.getds()[1:400,1])
print("\nTrain run succesfully\n")
print("________________________________________")

#Test for function setK in kdTreeKNNClassifier
print("\nFollowing is the test for kdTreeKNNClassifier setK Function, set k as 7 for example:\n")
print('knnkdt.setK(7)')
knnkdt.setK(7)
print("\nsetK run succesfully\n")
print("________________________________________")

#Test for function test in kdTreeKNNClassifier
print("\nFollowing is the test for kdTreeKNNClassifier test Function for continuous Data, output 20 predictions for example:\n")
print('knnkdt.test(qlds.getds()[401:,2:])[:19]')
print(knnkdt.test(qlds.getds()[401:,2:])[:19])
print("\nTest run succesfully, test data is saved\n")
print("________________________________________")

#Test for toString function in kdTreeKNNClassifier
print("\nFollowing is the test for kdTreeKNNClassifier toString Function:\n")
print('knnkdt.toString()')
print(knnkdt.toString())
print("\ntoString run successfully.\n")
print("________________________________________")

#Test for drawPrediction function in kdTreeKNNClassifier
print("\nFollowing is the test for kdTreeKNNClassifier drawPrediction Function:\n")
print('knnkdt.drawPrediction()')
knnkdt.drawPrediction()
print("\ndrawPrediction run successfully. One bar graph of plot is opened\n")
print("\nTest for kdTreeKNNClassifier run successfully\n")
print("________________________________________")
print("\nThat's the end of the test of kdTreeKNNClassifier Object\n")
print("========================================")
