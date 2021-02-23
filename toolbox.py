##* toolbox.py
##*
##* ANLY 555 Fall 2020
##* Project Deliverable5
##*
##* Due on: Nov 25, 2020
##* Authors: Leilin Wang, Shengdan Jin, Yifan Zhu, Jieqiao Luo
##*
##*
##* In accordance with the class policies and Georgetown's
##* Honor Code, I certify that, with the exception of the
##* class resources and those items noted below, I have neither
##* given nor received any assistance on this project other than
##* the TAs, professor, textbook and teammates.
##*

################################################################

# import related packages
import csv
import numpy as np
###
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import string
import re
import math
from collections import Counter
###
import matplotlib.pyplot as plt
import plotly.offline as py  
import plotly.graph_objs as go   
import plotly.express as px 
import plotly.figure_factory as ff
import time
###
#import animation_timeseries
#import animation_quantitydata
###
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
###
from random import randrange
import itertools

## Superclass DataSet
class DataSet:  
    
    ##constructor for DataSet
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @param dataset Orginial numpy array to set as dataset
    def __init__(self, filename,dataset=None):
        #set the attributes in the constructor
        
        if filename != "":
            ##attribute ds for saving dataset 
            # using function load to get ds
            self.ds = self.__load(filename)
            
        else:
            self.ds=dataset
        
        
        ##attribute file for saving filename 
        self.file = filename
        
        ##attribute rowcount for saving the how many rows in the dataset
        self.rowcount=self.ds.shape[0] - 1
        
        ##attribute columncount for saving the how many columns in the dataset
        self.columncount=self.ds.shape[1]     
        

    ## loadDataset function to ask type and load dataSet
    #  @param self The object pointer
    #  @return the dataset object with the type chosen
    def loadDataset(self):
        # ask for input and determine the type by if statement, and return the new object
        type = input("What is the type of the dataset? Enter TS for time series, TX for text, QL for qual, QN for quant: ")
        if type == "TS":
            return TimeSeriesDataSet(self.file)
        elif type == "TX":
            return TextDataSet(self.file)
        elif type == "QL":
            return QualDataSet(self.file)
        elif type == "QN":
            return QuantDataSet(self.file)
        else: 
            print ("you enter the wrong type")
            return DataSet(self.file)
        
    
    ##readsfromCSV function to read the dataset from csv file
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @return the dataset read with type np.array
    def __readsfromCSV(self, filename):
        # return the dataset read by csv package
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        
        return np.array(data)
        
    ##load function to detect if the file is csv and load the dataset from the filename, 
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @return the dataset loaded
    def __load(self, filename):

        #check if the dataset is csv 
        if (filename.endswith(".csv")):
            #if the dataset is csv, run readsfromCSV(filename)
            try:
                #try running readsfromCSV to set dataset to return dataset
                return self.__readsfromCSV(filename)
            # if the 
            except ImportError:
                print("The csv file does not exist")
        else:
            # print error message if the file name enter is not csv file
            print("Enter a csv file to load data set")
    
        #else if (filename.endswith()):
            #if it is not csv file, an example of read pds file is given and will be modify later

            #reurn tabula.read_pds(filename, encoding='utf-8', spreadsheet=True, pages='1-6041')
            
                
    ##clean function to do the data cleaning by removing all NA rows, need to override in subclass
    #  @param self The object pointer
    def clean(self):
        #print the dataset size before cleaning
        print("****"+"Before Cleaning")
        self.summary()
        #for clean in superclass, remove na by iterate over the numpyarray
        # narowlist variable to store na row index
        narowlist = []
        # count variable to calculate how many rows removed
        count = 0
        for x in range(1,self.rowcount):
            for y in self.ds[x,:]:
                # iterate each value to see whether the value is null
                if y == "":
                    # if null, add to narowlist and count and break it
                    count+=1
                    narowlist.append(x)
                    break
                
        # use LIFO to remove na rows in self.ds
        while narowlist:
            row = narowlist.pop()
            self.ds = np.delete(self.ds,row,0)
        
        print("****"+str(count)+" rows have been cleaned"+"****")
            
        
        #reset row and column after cleaning
        self.rowcount=self.ds.shape[0] - 1
        self.columncount=self.ds.shape[1]  

        #print the dataset size after cleaning
        print("****"+"After Cleaning")
        self.summary()
        
    ##explore function to draw visualization
    #  @param self The object pointer
    def explore(self):
        #visualization are determined by dataset type, so no graph will output in DataSet object
        print("Please specific your dataset type object for visualization")
    
    ##summary function for output the some identities of the dataset
    #  @param self The object pointer
    def summary(self):
        #print the size of the dataset
        print("****"+"The shape of dataset is " + str(self.rowcount) + " rows times " + str(self.columncount) +" columns"+"****")
    
    ##setColumnName function to set a column name to one column given column name, no need to override
    #  @param self The object pointer
    #  @param colNameOld The old colum name want to change
    #  @param colName The colum name need to be set to
    def setColumnName(self,colNameOld, colName):
        #try to set the column name
        try:
            self.ds[0,:] = [colName if x==colNameOld else x for x in self.ds[0,:]]
            #print the set result
            print( self.ds[0,:])
        except:
            #error message
            print("Column name not found")

    ##getColumn function to get a column by column name
    #  @param self The object pointer
    #  @param colName The colum name of column need to be return
    #  @param header for whether the header should be return or not
    #  @return column of corresponding column name
    def getColumn(self,colName,header = False):
        try:
            # loop to search the index of corresponding column name
            for i in range(0,self.columncount):
                if self.ds[0,i] == colName:
                    if header:
                        return self.ds[:,i]
                    else:
                        return self.ds[1:,i]
                
        except:
            #error message
            print("Column name not found")

            
    ##getColumnName function to get the list of column name
    #  @param self The object pointer
    #  @return List of column name
    def getColumnName(self):
        # return the list of column name
        return self.ds[0,:]
        
    #printDS function to print the dataset, no need to override
    #  @param self The object pointer
    def printDS(self):
        # return the string of dataset using to_string function
        print(self.ds)
    
    ##savetoCSV function to save the dataset to csv form
    #  @param self The object pointer
    #  @param filename The file name dataset save fort
    def savetoCSV(self, filename):
        # save the dataset to csv
        np.savetxt(filename, self.ds, delimiter=",",fmt='%s')
 
    ##head function to print the head dataset, no need to override
    #  @param self The object pointer
    def head(self):
        # print the head of dataset by printing first 6 rows
        print(self.ds[0:6,:])
        
    ##getds function to return dataset as numpy type
    #  @param self The object pointer
    #  @param header for whether the header should be return or not
    #  @return The dataset 
    def getds(self,header = False):
        if header:
            return self.ds
        else:
            return self.ds[1:,:]
    
    ##getUniqueCount function to return unique count of a column
    #  @param self The object pointer
    #  @param colName The colum name of column need to be return
    #  @return unqiue elements list, count elements list
    def getUniqueCount(self,colName):
        return np.unique(self.getColumn(colName),return_counts=True)
    
    ##removeColumn function to return a new dataset object with the column removed
    #  @param self The object pointer
    #  @param colName The colum name of column need to be remove
    #  @return a new dataset object
    def removeColumn(self,colName):
        datasettype=type(self)
        try:
            for i in range(0,self.columncount):
                if self.ds[0,i] == colName:
                    newds = np.delete(self.ds,i,1)
        except:
            #error message
            print("Column name not found")
        return datasettype("",dataset=newds)
    
################################################################

    
## Subclass TimeSeriesDataSet inherit from DataSet
class TimeSeriesDataSet(DataSet): 
    
    ##constructor for TimeSeriesDataSet inherits Dataet Class constructor
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @param dataset Orginial numpy array to set as dataset
    def __init__(self, filename,dataset=None):
        
        #inheritance from the DataSet constructor
        super().__init__(filename,dataset)
                
        ##attribute period for the specific period user want to analysis              
        self.period = 0     # set it to 0 if setPeriod method is not used     
        
        ##attribute ds_type default as original dataset
        self.ds_type= self.ds.copy()
        
        ##attribute filt_ds default as original ds_type
        self.filt_ds=self.ds_type.copy()
        
        self.period_start_index = 0
        self.period_end_index = self.columncount
        
        self.period_ds = self.ds.copy()
        
        
    ##changeType function to change time series type 
    #  @param self The object pointer
    #  @param colInd The colum index for getting column name
    #  @param timeType The type of time user changed to
    def changeType(self,colInd,timeType): 
        print("The column index for the time series type you want to change is "+str(colInd))       # confirm the column index chose by user
        print("The type of the time series (0 -- No change on current type; D -- Daily ; M -- Monthly ; Y -- Yearly) you want to change to is "+timeType)   # confirm the type of time series data that user want to change to
        
        self.ds_type = self.period_ds.copy()          
        
        if timeType == '0':                                  # keep the original type of time 
            return self.ds_type
        
        if timeType == 'D':                                  # change/resample to Daily data
            for i in range(len(self.ds_type[1:,colInd])):
                self.ds_type[1:,colInd][i] = np.datetime64(self.period_ds[1:,colInd][i],'D')
            return self.ds_type
        if timeType == 'M':                                # change/resample to Monthly data
            for i in range(len(self.ds_type[1:,colInd])):
                self.ds_type[1:,colInd][i] = np.datetime64(self.period_ds[1:,colInd][i],'M')                              
            return self.ds_type
        if timeType == 'Y':                                  # change/resample to Yearly data
            for i in range(len(self.ds_type[1:,colInd])):
                self.ds_type[1:,colInd][i] = np.datetime64(self.period_ds[1:,colInd][i],'Y') 
            return self.ds_type
        else:
            print("timeType: Please choose from '0' -- Original type of time,'D' -- (Daily), 'M' -- (Monthly) or 'Y' -- (Yearly).")

    
    ##override clean function to do the data cleaning by running a median filter
    #  @param self The object pointer
    #  @param filter_size The size of median filter window
    #  @return filt_ds Cleaned(median filtered) dataset
  
    def clean(self,filter_size):        
        if filter_size % 2 == 0:  
            print("The filter size must be an ODD number.")       # filter size must be an odd number
 
        else:
            med_index = (filter_size -1 ) // 2                    # first get an empty array to contain the value of filtered data
            noheaderds = self.ds_type[1:].copy()
            self.filt_ds = self.ds_type.copy()
            
            for i in range (1,len(noheaderds[0])):                # apply Median Filtering to smooth the data
                for j in range (med_index, len(noheaderds)-med_index): 
                    self.filt_ds[j,i] = np.median(noheaderds[j-med_index:j+med_index+1,i].astype(np.float))
     
            return self.filt_ds                                        # return cleaned(median filtered) dataset 
        


    ##override explore function to draw visualization.
    ##plot line chart for chosen column data before and after filtering 
    #  @param self The object pointer
    #  @param colName Column want to explore
    def explore(self, colName):
        try:
            for i in range(0,self.columncount):
                if self.ds[0,i] == colName:
                    colInd = i
        except:
            #error message
            print("Column name not found")
            
        facts=self.filt_ds    # filtered data
        
        
        facts2=self.ds_type       # original data

        list2_name=self.period_ds[1:,0]
        
        trace0 = go.Scatter(      # plot filtered column of ColName
              x = list2_name,
              y = facts[1:,colInd],
              mode = 'lines',
              name = colName+"_filter"
        )
                
        trace1 = go.Scatter(      # plot original column of ColName
              x = list2_name,
              y = facts2[1:,colInd],
              mode = 'lines',
              name = colName   
        )
        
        
        data = [trace0, trace1] # before and after filtering 
        py.plot(data) #plot 
        print("Show the line plot for ",colName, " price before and after filtering" )
        
        time.sleep(5)
        fig = go.Figure(data=[go.Candlestick(    x=self.period_ds[1:,0],
                                            open=self.period_ds[1:,1].astype(np.float64), high=self.period_ds[1:,2].astype(np.float64),
                                            low=self.period_ds[1:,3].astype(np.float64), close=self.period_ds[1:,4].astype(np.float64),
                                            increasing_line_color= 'cyan', decreasing_line_color= 'gray')])

        py.plot(fig)
        
    
    ##override summary function for output the some identities of the TimeSeriesDataset. 
    ##It shows data information and description
    #  @param self The object pointer
    def summary(self):
           
        #add summary of time series data
        print("The dimension of filtered array is:", self.filt_ds.shape)


      
    ##setPeriod function to set the period value you want choose for stock price
    #  @param self The object pointer
    #  @param start The start of period user want to set 
    #  @param end The end of period user want to set 
    #  @return PeriodData() result
    def setPeriod(self,start,end):
        #set the attribute period to the period user input
        self.period_start_index = (np.where(self.ds == start))[0][0] # get the row index of start of period 
        self.period_end_index = (np.where(self.ds == end ))[0][0]    # get the row index of the end of peroid
        
        
        print("The period you set is", start, "to", end) # confirm the period set by user
        return self.PeriodData()
    
    ##periodData function to return the period data
    #  @param self The object pointer
    #  @return dataset modify by period
    def PeriodData(self):
        self.period_ds=self.ds[np.r_[0, self.period_start_index:(self.period_end_index+1)]]
        return self.period_ds

    ##sort dataframe for date column
    #  @param self The object pointer
    #  @return Dataset
    def sort(self):
        try:
            Columndata=self.getColumn("Date")
            


            index=np.argsort(Columndata.astype(np.datetime64))
            
            Sorted_data=self.ds[index+1,:]
            self.ds=np.vstack((self.getColumnName(),Sorted_data))
            return self.ds
        except:
            print('Fail to sort dataframe, Date column exists inappropriate value')
        

        
################################################################


## Subclass TextDataSet inherit from DataSet
class TextDataSet(DataSet): 
    
    ##constructor for TextDataSet inherits Dataet Class constructor
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @param dataset Orginial numpy array to set as dataset
    def __init__(self, filename,dataset=None):
         
        #inheritance from the DataSet constructor
        super().__init__(filename,dataset)
        ## attribute step_words to get stopwords
        self.stop_words=set(stopwords.words('english')) 
        ## attribute facts_texts to get the text column from the dataset
        self.facts_texts=self.getColumn('text')
        ## attribute texts to get all the texts 
        self.texts=[]
        for i in self.facts_texts:
            self.texts.append(i)
        
        

    ##override clean function to do the data cleaning, like stopwords removing
    #  @param self The object pointer
    def clean(self):
        
        # clean and tokenize the sentences
        filtered_texts=[]
        
        # for all the text s
        for text in self.texts:
            word_tokens = word_tokenize(text) 
  
            filtered_text = [w for w in word_tokens if not w in self.stop_words] 
            filtered_text = [" ".join(filtered_text)]

            filtered_texts.append(filtered_text) 
        
        return filtered_texts
    
    ##override explore function to draw visualization, like wordcloud and star rating bar chart
    #  @param self The object pointer
    def explore(self):
        # generating a text from 
        
        texts_for_wordcloud=''
        
        for i in self.texts:
            texts_for_wordcloud+=i
            
        wcloud = WordCloud(
            background_color='white',
            #font_path="/System/Library/AssetsV2/com_apple_MobileAsset_Font6/fedef1896002be99406da1bf1a1a6104a1737b39.asset/AssetData/Xingkai.ttc",
            max_words=2000, 
            stopwords=self.stop_words,
            max_font_size=150,
            random_state=30
        )
        
        wcloud.generate(texts_for_wordcloud)
        
        
        plt.imshow(wcloud)
        
        plt.axis('off')
        plt.show()
        
        
        star_key,star_value= np.unique(self.getColumn('stars'), return_counts=True)
         
        
        time.sleep(5)
        fig = px.bar(self.getColumn('stars'), x=star_key, y=star_value,
              barmode='group',
             height=400)
        py.plot(fig)     
        
    ##getText function to return the text label
    #  @param self The object pointer
    #  @return text label
    def getText(self):
        return self.texts
        
        
    ##remove puncutation for text data and update into original text data
    #  @param self The object pointer
    #  @return text label
    def remove_punctuation(self):        
        removed_punctuation_list=[]
        for text in self.texts:
            no_punct = "".join([c for c in text if c not in string.punctuation])
            removed_punctuation_list.append(no_punct)
        self.texts=removed_punctuation_list
        return self.texts
    
    ##remove number for text data and update into original text data
    #  @param self The object pointer
    #  @return text label
    def remove_number(self):
        for i in range(len(self.texts)):
            self.texts[i]= re.sub(r'[0-9]+', '', self.texts[i])
            
        return self.texts
    
    ##tokenize every row for text data and create a tokenized list    
    #  @param self The object pointer
    #  @return tokenized list
    def tokenize (self):
        
        tokenize_list=[]
        for text in self.texts:
            tokenize_list.append(nltk.word_tokenize(text))
        
        return tokenize_list         


    ##Stemmers remove morphological affixes from words, leaving only the word stem and update into original text data  
    #  @param self The object pointer
    #  @return text label
    def stemming(self):
        tokneized_text=self.tokenize()
        ps=PorterStemmer()
        stemm_list=list()
        for text in tokneized_text:
            stemmed = " ".join([ps.stem(word) for word in text])
            stemm_list.append(stemmed)
        self.texts=stemm_list
        return self.texts
    
    ##Implement lemmatization to groupe together the different inflected forms of a word and update into original text data
    #  @param self The object pointer
    #  @return text label
    def lemmatization(self):
        tokneized_text=self.tokenize()
        lemmatizer = WordNetLemmatizer() 
        lemmatize_list=list()
        for text in tokneized_text:
            lemmatized= " ".join([lemmatizer.lemmatize(word) for word in text])
            lemmatize_list.append(lemmatized)
        self.texts=lemmatize_list
        return self.texts 
   
################################################################


## Subclass QuantDataSet inherit from DataSet
class QuantDataSet(DataSet): 
    
    ##constructor for QuantDataSet inherits Dataet Class constructor
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @param dataset Orginial numpy array to set as dataset
    def __init__(self, filename,dataset=None):
        
        #inheritance from the DataSet constructor
        super().__init__(filename,dataset)
        
        

    ##override clean function to do the data cleaning, like filling in missing values with the mean, if possible, if no mean can be generated, remove all na rows
    #  @param self The object pointer
    def clean(self):
        #print the dataset size before cleaning
        print("Before Cleaning")
        super().summary()
        
        mean = []
        # first calculate mean to fill the na row, if mean cannot calculate, delete na rows
        for x in self.getColumnName():
            try:
                mean.append(np.mean(self.getColumn(x).astype(np.float)))
            except:
                mean.append("")

        # narowlist variable to store na row index
        narowlist = []
        # count variable to calculate how many rows removed
        count = 0
        navaluefilled = 0
        for x in range(1,self.rowcount):
            for y in range(0,self.columncount):
                # iterate each value to see whether the value is null
                if self.ds[x,y] == "":
                     # if null and have mean:
                    if mean[x] != "":
                        self.ds[x,y] = mean[x]
                        navaluefilled +=1
                    else:
                    # if null and do not have mean
                        count+=1
                        narowlist.append(x)
                        break
                
        # use LIFO to remove na rows in self.ds
        while narowlist:
            row = narowlist.pop()
            self.ds = np.delete(self.ds,row,0)
        
        print(str(count)+" rows have been cleaned")
        print(str(navaluefilled)+" value have been replaced")  
        
        #reset row and column after cleaning
        self.rowcount=self.ds.shape[0] - 1
        self.columncount=self.ds.shape[1]  

        #print the dataset size after cleaning
        print("After Cleaning")
        super().summary()
        
    ##override explore function to draw visualization
    ##bar chart to show every weekly sales for each product
    ##line chart to show input product' sales
    #  @param self The object pointer
    #  @param product_number The product number of figure one
    #  @param weekName The week name of figure two
    def explore(self,product_number,weekName):
        
        week_list=list(self.getColumnName()[1:53])
        
        #animation_quantitydata.SaleBarChart(week_list)

        #product_name=input("Please input product name that you want to get the sales line chart like P1 or P2:")
        #product_name=list(product_name)
        #product_number=int(product_name[1])
        
        time.sleep(5)
        trace = go.Scatter(
              x = week_list,
              y = self.ds[product_number,1:53],
              mode = 'lines',
              name = 'Volumn'   
        )
        data = [trace]
    
        py.plot(data)        
        
        time.sleep(5)
        fig2 = px.bar([self.getColumn(weekName),self.getColumn("Product_Code")], x=self.getColumn("Product_Code"),y=self.getColumn(weekName)) 
        py.plot(fig2)  
        
    ##override summary function for output the some identities of the QuantDataSet
    #  @param self The object pointer
    #  @param colInd Column Index for column want to output the summary
    def summary(self,colInd):
        #use the summary function in DataSet Class
        super().summary()
        
        #print min, max, mean, median as summary for quantdataset
        for x in self.getColumnName()[colInd]:
            try:
                print("\nFor column "+ x)
                print("Mean is " + str(np.mean(self.getColumn(x).astype(np.float))))
                print("Min is " + str(np.min(self.getColumn(x).astype(np.float))))
                print("Median is " + str(np.median(self.getColumn(x).astype(np.float))))
                print("Max is "+ str(np.max(self.getColumn(x).astype(np.float))))
            except:
                print("No summary can be print")
                
    ##sort dataframe for one selected ColName
    #  @param self The object pointer
    #  @param ColName The ColumnName you want to sort dataframe according to that column
    #  @return Dataset
    def sort(self,ColName):
        try:
            Columndata=self.getColumn(ColName)
            
            index=np.argsort(Columndata.astype(np.float))
            
            Sorted_data=self.ds[index+1,:]
            self.ds=np.vstack((self.getColumnName(),Sorted_data))
            return self.ds
        except:
            print('Fail to sort dataframe, column name is wrong or there is value that can not be formed to float type')

    ##getColumn function to get a column by column name
    #  @param self The object pointer
    #  @param colName The colum name of column need to be return
    #  @param header for whether the header should be return or not
    #  @return column of corresponding column name
    def getColumn(self,colName,header = False):
        try:
            # loop to search the index of corresponding column name
            for i in range(0,self.columncount):
                if self.ds[0,i] == colName:
                    if header:
                        return self.ds[:,i]
                    else:
                        return self.ds[1:,i]
                
        except:
            #error message
            print("Column name not found")

            
    ##getColumnName function to get the list of column name
    #  @param self The object pointer
    #  @return List of column name
    def getColumnName(self):
        # return the list of column name
        return self.ds[0,:]    
    
################################################################

## Subclass QualDataSet inherit from DataSet
class QualDataSet(DataSet): 
    
    ##constructor for QualDataSet inherits Dataet Class constructor
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @param dataset Orginial numpy array to set as dataset
    def __init__(self, filename,dataset=None):
        
        #inheritance from the DataSet constructor
        super().__init__(filename,dataset)
        
    ##override clean function to do the data cleaning, filling in missing values with the mode
    #  @param self The object pointer
    def clean(self):
        

        #print the dataset size before cleaning
        print("Before Cleaning")
        super().summary()
        
        mode = []
        # first calculate mode to fill the na row, if mean cannot calculate, delete na rows
        for x in self.getColumnName():
             (values,counts) = np.unique(self.getColumn(x),return_counts=True)
             ind=np.argmax(counts)
             mode.append(values[ind])

        navaluefilled = 0
        for x in range(1,self.rowcount):
            for y in range(0,self.columncount):
                # iterate each value to see whether the value is null
                if self.ds[x,y] == "":
                    self.ds[x,y] = mode[y]
                    navaluefilled +=1

        print(str(navaluefilled)+" value have been replaced")  
        
        #reset row and column after cleaning
        self.rowcount=self.ds.shape[0] - 1
        self.columncount=self.ds.shape[1]  

        #print the dataset size after cleaning
        print("After Cleaning")
        super().summary()

    ##override explore function to draw visualization
    ##pie chart is used to show selected column's distribution among nationalities
    #bar chart is used to show selected columns' distributions
    #  @param self The object pointer
    #  @param column_name The column name for pie chart
    #  @param x_axis The column name for x axis of bar chart
    #  @param color_axis The column name for color of bar chart
    def explore(self,column_name,x_axis,color_axis):
        
        #create a pie chart of colName for Nationality distribution
        #column_name=input("Please input the column name you want to show its distribution for Nationality by using pie chart(For example Nationality, Age):")
        time.sleep(5)
        fig1 = px.pie(self.ds, names=self.getColumn(column_name), title=column_name)
        py.plot(fig1)
     
        
        #create a bar chart for two columns, one is x axis, the other is color filling
        time.sleep(5)
        #print("Next, a bar chart will be created by inputting ")
        self.fact_value=self.ds[1:,:]
        #x_axis=input("Please input the column name you want to create a bar chart as x axis(For example Nationality, Age):")
        #color_axis=input("Please input the column name you want to use as color filling for the bar chart(For example Nationality, Age):")
        fig2 = px.bar(self.fact_value, x=self.getColumn(x_axis),color=self.getColumn(color_axis)) 
        py.plot(fig2)   
        
    ##override summary function for output the some identities of the QualDataSet
    #  @param self The object pointer
    def summary(self):
        #use the summary function in DataSet Class
        super().summary()
        
        
        #print mode and unique count as summary for quantdataset
        for x in self.getColumnName():
            try:
                #print out mode
                (values,counts) = np.unique(self.getColumn(x),return_counts=True)
                ind=np.argmax(counts)
                mode=values[ind]
 
                #loop to print out unique value, if value occurs more than 2 times
                if counts[ind] >= 3:
                    print("\nFor column "+ x)
                    print("Mode is " + str(mode))
                    for i in range(0,len(values)):
                        print("Value \'"+str(values[i]) + "\' occurs " + str(counts[i])+" times")
            except:
                print("No summary can be print")
                

    ##getColumn function to get a column by column name
    #  @param self The object pointer
    #  @param colName The colum name of column need to be return
    #  @param header for whether the header should be return or not
    #  @return column of corresponding column name
    def getColumn(self,colName,header = False):
        try:
            # loop to search the index of corresponding column name
            for i in range(0,self.columncount):
                if self.ds[0,i] == colName:
                    if header:
                        return self.ds[:,i]
                    else:
                        return self.ds[1:,i]
                
        except:
            #error message
            print("Column name not found")

            
    ##getColumnName function to get the list of column name
    #  @param self The object pointer
    #  @return List of column name
    def getColumnName(self):
        # return the list of column name
        return self.ds[0,:]    
    
    ##removeColumn function to return a new dataset object with the column removed
    #  @param self The object pointer
    #  @param colName The colum name of column need to be remove
    #  @return a new dataset object
    def removeColumn(self,colName):
        datasettype=type(self)
        try:
            for i in range(0,self.columncount):
                if self.ds[0,i] == colName:
                    newds = np.delete(self.ds,i,1)
        except:
            #error message
            print("Column name not found")
        return datasettype("",dataset=newds)
    
################################################################

## Subclass TransactionDataSet inherit from DataSet
class TransactionDataSet(DataSet):
    
    ##constructor for TransactionDataSet inherits Dataet Class constructor
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @param dataset Orginial numpy array to set as dataset
    def __init__(self, filename,dataset=None):
        
        if filename != "":
            ##attribute ds for saving dataset 
            # using function load to get ds
            self.ds = self.__load(filename)
            
        else:
            self.ds=dataset

        


    ##readsfromCSV function to read the dataset from csv file
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @return the dataset read with type np.array
    def __readsfromCSV(self, filename):
        # return the dataset read by csv package
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        
        return np.array(data)
    
    ##load function to detect if the file is csv and load the dataset from the filename, 
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @return the dataset loaded
    def __load(self, filename):

        #check if the dataset is csv 
        if (filename.endswith(".csv")):
            #if the dataset is csv, run readsfromCSV(filename)
            try:
                #try running readsfromCSV to set dataset to return dataset
                return self.__readsfromCSV(filename)
            # if the 
            except ImportError:
                print("The csv file does not exist")
        else:
            # print error message if the file name enter is not csv file
            print("Enter a csv file to load data set")
    
        #else if (filename.endswith()):
            #if it is not csv file, an example of read pds file is given and will be modify later

            #reurn tabula.read_pds(filename, encoding='utf-8', spreadsheet=True, pages='1-6041')
    
    
    ##clean function to change the form of transaction data so each item and each set can be available for ARM
    #  @param self The object pointer 
    #  @param filename The file name dataset read from   
    def clean(self):
        ##attribute trans to contain transaction data in form of bracket data
        self.trans = []
        for line in self.ds:
            str_line = list(line[0].strip().split(','))
            tr = list(np.unique(str_line))
            self.trans.append(tr)
        print("The dataset before cleaning:")
        print('--------------------------------------------------')
        print(self.ds)
        print()
        print("The dataset after cleaning/tranforming into the form that each item and each set could be studied for frequent association rules :")
        print('--------------------------------------------------')
        print(self.trans)
            

                

    ##explore function to print out top 10 association rules above supportThreshold with computed support, confidence and lift
    #  @param self The object pointer
    def explore(self):
        return self.__ARM__() #call __ARM__ method
    
    

    
    ##FrequentItem function to print out all frequent item sets
    #  @param self The object pointer
    #  @param supportThreshold The minimal support that will be used to discard infrequent rules
    def FrequentItem(self,supportThreshold=0.25): 
        ##attribute self.apriori to get oranized data through Rule class
        self.apriori = self.Rule(self.trans,supportThreshold)  # call the Rule class
        
        freqSet = self.apriori.generate_L()
        for Lk in freqSet:
            print('--------------------')
            print('* Frequent {}-itemsets：\n'.format(len(list(Lk)[0])))
            print('--------------------')

            for freq in Lk:
                print(set(freq), 'support:', self.apriori.support_data[freq])
    
            print()
    
    

    
    ##ARM function to generate top 10 association rules above supportThreshold with computed support, confidence and lift
    #  @param self The object pointer
    def __ARM__(self):
        ##attribute self.apriori to get oranized data through Rule class
        self.apriori = self.Rule(self.trans,supportThreshold=0.15)  # call the Rule class
        
        rule_list = self.apriori.generate_rules()
        
        print("Top 10 rules:")
        print('----------------------------------')
        for item in rule_list:
            print(item[0], "=>", item[1], "support: ", item[2], "confidence:", item[3], "lift:",item[4])
        print()
        
        
        print("Top 10 rules -- Support:")
        print('----------------------------------')
        for item2 in rule_list:
            print(item2[0], "=>", item2[1], "support: ", item2[2])
        print()

        print("Top 10 rules -- Confidence:")
        print('----------------------------------')
        for item in rule_list:
            print(item[0], "=>", item[1], "confidence: ", item[3])
        print()
        
        print("Top 10 rules -- Lift:")
        print('----------------------------------')
        for item1 in rule_list:
            print(item1[0], "=>", item1[1], "lift: ", item1[4])
        print()
    
    ##summary function prints out the total number of rules being generated 
    #  @param self The object pointer
    def summary(self):
        print("There are in total", self.apriori.rule_list_length, "association rules generated with only top 10 frequent rules printed." )
            
    ## Innerclass Rule to organize data for ARM        
    class Rule:
        
        ##constructor for Rule
        #  @param self The object pointer
        #  @param transactions The transaction dataset that will be used for ARM
        #  @param supportThreshold The minimal support that will be used to discard infrequent rules
        def __init__(self, transactions, supportThreshold):
            self.transactions = transactions
            self.supportThreshold = supportThreshold # The minimum support.
            self.support_data = {} # first create an empty dictionary to contain key:frequent itemset and value: support.
            
        ##candidate_1_item function to create all frequent candidate 1-itemsets (each set includes 1 item that appears frequently among all the transaction) 
        #  @param self The object pointer
        #  @return A set which contains all frequent candidate 1-itemsets
        def candidate_1_item(self):
    
            C1 = set()   # first create an empty set to contain all frequent candidate 1-itemsets
            # Add all frequent candidate 1-itemsets to C1
            for transaction in self.transactions:
                for item in transaction:
                    C1.add(frozenset([item]))
            return C1          
        
        
        ##candidate_k_item function to create all frequent candidate k-itemsets (each set includes k items that appear frequently among all the transaction) 
        #  @param self The object pointer
        #  @param last_iteration A set containing all frequent (k-1)-itemsets 
        #  @param k The item number of a frequent itemset
        #  @return A set which contains all frequent candidate k-itemsets
        def candidate_k_item(self, last_iteration, k):
    
            Ck = set()  # first create an empty set to contain all frequent candidate k-itemsets
            len_last_iteration = len(last_iteration)
            list_last_iteration = list(last_iteration)
            for i in range(len_last_iteration):
                for j in range(i+1, len_last_iteration):
                    l1 = list(list_last_iteration[i])
                    l2 = list(list_last_iteration[j])
                    l1.sort()
                    l2.sort()
                    if l1[0:k-2] == l2[0:k-2]:
                        # self joining the last iteration containing all frequent (k-1)-itemsets
                        Ck_tmp = list(set(l1) | (set(l2)))
                        # pruning
                        flag = 1
                        for k in range(len(Ck_tmp)):
                            tmp = Ck_tmp.copy()
                            tmp.pop(k)
                            if not set(tmp) in last_iteration:
                                flag = 0
                                break
                        if flag:
                            Ck.add(frozenset(Ck_tmp))
            return Ck
        
        
        
        ##Iteration_k_from_candidate_k function to create Iteration which contains frequent k-itemsets by executing a delete policy from frequent candidate k-itemsets
        #  @param self The object pointer
        #  @param Ck A set which contains all frequent candidate k-itemsets
        #  @return A set which contains all frequent k-itemsets
        def Iteration_k_from_candidate_k(self, Ck):
            
            Lk = set()
            item_count = {}
            for transaction in self.transactions:
                for item in Ck:
                    if item.issubset(transaction):
                        if item not in item_count:
                            item_count[item] = 1
                        else:
                            item_count[item] += 1
            t_num = float(len(self.transactions))               # get total number of transactions
            for item in item_count:
                support = item_count[item] / t_num    # compute support
                if support >= self.supportThreshold:
                    Lk.add(item)
                    self.support_data[item] = support
            return Lk
        
        
        
        
           
        ##generate_L function to create all frequent item sets
        #  @param self The object pointer
        #  @return A list of the sets of all frequent k-itemsets
        def generate_L(self):
       
            self.support_data = {}
            
            C1 = self.candidate_1_item()
            L1 = self.Iteration_k_from_candidate_k(C1)
            last_iteration = L1.copy()
            L = []
            L.append(last_iteration)
            i = 2
            while True:
                Ci = self.candidate_k_item(last_iteration, i)
                Li = self.Iteration_k_from_candidate_k(Ci)
                if Li:
                    last_iteration = Li.copy()
                    L.append(last_iteration)
                    i += 1
                else:
                    break
            return L
        
        
     
      
    
        
        ##generate_rules function to generate top 10 association rules from frequent itemsets
        #  @param self The object pointer
        #  @return Top 10 rules with support, confidence and lift
        def generate_rules(self):
            
            L = self.generate_L()
            
            top_rule_list = []
            sub_set_list = []
            for i in range(0, len(L)):
                for freq_set in L[i]:
                    for sub_set in sub_set_list:
                        if sub_set.issubset(freq_set):
                            # compute the support
                            sup = float(format(self.support_data[freq_set],'.3f'))
                            conf = float(format(self.support_data[freq_set] / self.support_data[freq_set - sub_set],'.3f'))
                            lift = float(format(self.support_data[freq_set] / ((self.support_data[freq_set - sub_set])*(self.support_data[sub_set])),'.3f'))
                            top_rule = (set(freq_set - sub_set), set(sub_set), lift)
                            top_rule = (set(freq_set - sub_set), set(sub_set), sup, conf, lift)
                            if top_rule not in top_rule_list:
                                top_rule_list.append(top_rule)
                    sub_set_list.append(freq_set)
            self.rule_list_length = len(top_rule_list)
            return top_rule_list[:10]
                

################################################################         
 
## Superclass ClassifierAlgorithm       
class ClassifierAlgorithm:  
    
    ##constructor for ClassifierAlgorithm
    #  @param self The object pointer
    def __init__(self):      
        #set the attributes in the constructor
        
        ##attribute trainTestRatio indicates for train test ratio
        self.trainTestRatio=[0.7,0.3] # now initialize to 0.7 and 0.3, which means that 70% of data will be group to train set and 30% will be group to test set
        
        ##attribute perdicton indicates empty prediction result list to add result later
        self.prediction= []   
        
    ##splitTrainTest function to split a dataset to train and test set, train and test label
    #  @param self The object pointer
    #  @param dataset Dataset, numpy type array, pass in for splitting, header not included
    #  @param label Label, numpy type array, pass in for splitting, header not included
    #  @return the test set splitted by the ratio
                                                                               # **------ m: length of self.trainL ------**
    def splitTrainTest(self,dataset,label):                                    # 1 step
        trainindex = int(len(label) * self.trainTestRatio[0])                  # 5 steps: self.trainTestRatio[0], *, len(), int(),trainindex = 
        self.train(dataset[:(trainindex-1),],label[:(trainindex-1)])           # 4m+1+3 steps: 3 steps-- trainindex-1, dataset[], label[]; 4m+1 steps -- self.train()
        self.testDS=dataset[trainindex:,]                                      # 2 steps: dataset[], self.testDS=
        return self.testDS                                                     # 1 step
   
    # -----------------------------------------------------------------------
    # Total:
    # m: length of self.trainL 
    # T(m) = 4m + 13
    # T(m) is O(m) = m
    # -----------------------------------------------------------------------
    
    
    
    ##setratio function to set the train and test ratio
    #  @param self The object pointer
    #  @param trainratio Train ratio want to set for randomsplit
    def setRatio(self,trainratio):
        if trainratio<1 and trainratio>0:    
            self.trainTestRatio = [trainratio,1-trainratio]
        else:
            print("Invalid train ratio")
        
    ##train function to save train set
    #  @param self The object pointer
    #  @param trainData The training dataset
    #  @param labl Training label
    def train(self,trainData,label):
        #save parameter to trainDS and trainL
        ##attribute trainDS for saving training set
        self.trainDS = trainData                   # 1 step ; 1 space
        
        ##attribute trainL for saving training Label
        self.trainL = label                        # 1 step ; 1 space
        
        ##set prediction score attribute                                        # **------ m: length of self.trainL ------**
        unique_train, counts_train=np.unique(self.trainL, return_counts=True)   # m steps ; 3 space: unique_train, counts_train, np.unique()
        #train_label_count=dict(zip(unique_train, counts_train))
        self.prediction_score= [[] for _ in unique_train]                       # 3m+1 steps: _ =, _< length of unique_train, _++, [] -- 3m steps, self.prediction_score = [] -- 1 step ; 3 space: _, [], self.prediction_score

    # -----------------------------------------------------------------------
    # Total:
    # m: length of self.trainL 
    # T(m) = 4m + 1
    # T(m) is O(m) = m
    # S(m)= 6
    # S(m) is O(m) = 6
    # -----------------------------------------------------------------------

       
    ##test function to return test set
    #  @param self The object pointer
    #  @param testData The test dataset
    def test(self,testData):                            # 1 step
        ##attribute testDS for saving test set
        self.testDS = testData                          # 1 step
        
        #print("Test data is saved")
    
    ##toString function to return the name of classifier
    #  @param self The object pointer
    def toString(self):
        return "Classifier"
    
    ##testApplicable function to test whether the dataset is applicable for the classifier
    #  @param self The object pointer
    #  @param ds The data set pass in to test applicable
    #  @param l The label pass in to test applicable
    #  @return boolean whether the dataset is applicable
    def testApplicable(self,ds,l):
        print(self.toString()+ " is not applicable for this dataset")
        return False
    
    
    
    ##drawPrediction function to draw the result of prediction
    #  @param self The object pointer
    def drawPrediction(self):
        unique = sorted(set(self.prediction))
        count = []
        for p in unique:
            num = 0
            for i in self.prediction:
                if p ==i:
                    num+=1
            count.append(num)
        time.sleep(5)
        fig = px.bar(x=unique, y=count,labels=dict(x="Prediction Result",y="Count"), title = "Bar Chart for Prediction result")
        py.plot(fig)
        
################################################################
        
## Subclass simplekNNClassifier inherit from ClassifierAlgorithm
class simplekNNClassifier(ClassifierAlgorithm):  
    
    ##constructor for simplekNNClassifier
    #  @param self The object pointer
    def __init__(self):
        
        #inheritance from the ClassifierAlgorithm constructor
        super().__init__()
        
        ##attribute k that is saved for default
        self.k = 3
        

    ##override train function to save train set
    #  @param self The object pointer
    #  @param trainData The training dataset
    #  @param labl Training label
    def train(self, trainData, label):
        super().train(trainData,label)

    ##setK function to set k in the classifier type
    #  @param self The object pointer
    #  @param kvalue Value of k want to set
    def setK(self, kvalue):
        self.k = kvalue
        print("kvalue is saved")
    
    ##askK function to set k in the classifier type
    #  @param self The object pointer
    def askK(self):
        kvalue = input("What is the kvalue you want to save?")
        self.k=kvalue
        print("kvalue is saved")
    
    
   ##override test function to return test set
    #  @param self The object pointer
    #  @param testData The test dataset
    #  @param k parameter for kNN
    #  @return the prediction result of kNN, the prediction probability, Label of probability
    def test(self,testData,k=None):                 # 1 step
        
        super().test(testData)                      # 1 step
        
        if k == None:                               # 2 steps: if, ==
            k = self.k                              # 1 step, 1 space

        
        # calculate the Euclidean distance between two vectors  
                                                    # --------------------
        def euclidean_distance(row1, row2):         # 1 step
            distance = 0.0                          # 1 step, 1 space
            for i in range(len(row1)-1):            # 4n steps, 5 space: i, range, length, length-1 boolean
            	distance += (row1[i] - row2[i])**2  # 6n steps, 2 space: row1[i], row2[i]
            return (distance)**(1/2)                # 3 steps, 1 space 
                                                    # --------------------
                                                    # Total: T(n) = 10n+5, S(n) = 9
                                                    
        # empty prediction result list to add result later
        self.prediction= []                         # 1 step, 1 space
        
        
        unique_train, counts_train=np.unique(self.trainL, return_counts=True)
        train_label_count=dict(zip(unique_train, counts_train))
        self.prediction_score= [[] for _ in unique_train] 

        
        
        # for each data point in test data          # m: len(testData)
        for a in range(0,len(testData)):            # 2m+2 steps, 4 space: a, range, length, boolean 
            # empty distance list to add distance
            distances = []                          # m steps, 1 space
            
  
                                          
            # calculate the distance between test point and all train point
                                                    # --------------------
                                                    # h: len(self.trainDs)
            for b in range(0,len(self.trainDS)):    # 2h+2 steps, 4 space: b, range, length, boolean 
                distance = euclidean_distance(self.trainDS[b,:].astype(np.float), self.testDS[a,:].astype(np.float))
                                                    # (5+(10n+5))*h steps, 12 space
                # add the distance and its index into distances list
                distances.append([distance,b])      # h step
                                                    # --------------------
                                                    # Total: T(n,h) = h*(10n+11)+(2h+2), S(n,h) = 17
                                                    

            # sort the list
            sorteddist = sorted(distances,key=lambda x:x[0])  # 3+hlog(h) steps, 3 space
            
            knindex = []                                      # 1 step, 1 space
            
            # find the first k index
                                                              # --------------------
            for i in range(0,k):                              # 2k+1 steps, 3 space
                knindex.append(sorteddist[i][1])              # 2k steps, 1 space
                                                              # --------------------
                                                              # Total: T(k) = 4k+1, S(k) = 4
                
            # Find the mode of knn by index
            (values,counts) = np.unique(np.take(self.trainL,knindex),return_counts=True)  
                                                              # 5m steps, 5 space
                                                              
                                                              
                                                                      
            for i in range(len(train_label_count.keys())):
                Positive_location=np.where(values == list(train_label_count.keys())[i])                                     
                self.prediction_score[i].append(sum(counts[Positive_location].astype(np.float))/sum(counts))  

                
                
                
                
            ind=np.argmax(counts)                             # 2m steps, 1 space
            self.prediction.append(values[ind])               # 2m steps, 1 space

        return self.prediction,self.prediction_score,list(train_label_count)                                 # 1 step, 1 space
    
    
    ##testApplicable function to test whether the dataset is applicable for the classifier
    #  @param self The object pointer
    #  @param ds The data set pass in to test applicable
    #  @param l The label pass in to test applicable
    #  @return boolean whether the dataset is applicable
    def testApplicable(self,ds,l):
        try:
            self.test(self.splitTrainTest(ds,l),3)
            return True
        except:
            print(self.toString()+ " is not applicable for this dataset")
            return False
    
    # -----------------------------------------------------------------------
    # for loop:
    # Total:
    # T(k,h,n,m) = (4k+1)*m + (h*(10n+11)+(2h+2)+1 + 3 + hlog(h))*m + m + (2m+2) + 9m + 6
    # T(k,h,n,m) is O(k,h,n,m) = km + mhn + hm + mhlog(h)
    # S(k,h,n,m)= 50
    # S(k,h,n,m) is O(k,h,n,m) = 50
    # -----------------------------------------------------------------------
        
    ##toString function to return the name of classifier
    #  @param self The object pointer
    def toString(self):
        return "kNNClassifier with k as "+str(self.k)
        
 

################################################################


## super tree class
class Tree: 

    # buildTree function to insert nodes in level order  
    #  @param self The object pointer
    #  @param arr Array storing the data for tree
    #  @param root The originial input root
    #  @param i Used for base case of recursion
    #  @param n Used for base case of recursion
                                                              # **-- n: len(arr) --**
    def buildTree(self,arr, root, i, n):                      # 1 step
          
        # Base case for recursion  
        if i < n:                                             # 2 steps: if, <
            root = self.TreeNode(arr[i])                      # 3 steps: arr[i], self.TreeNode(), root = 
      
            # insert left child  
            root.left = Tree.buildTree(arr, root.left,        # T(n/2)+1 steps, Tree.buildTree(), root.left =
                                         2 * i + 1, n)  
      
            # insert right child  
            root.right =Tree.buildTree(arr, root.right,       # T(n/2)+1 steps, Tree.buildTree(), root.right =
                                          2 * i + 2, n) 
        return root                                           # 1 step
    
    # -----------------------------------------------------------------------
    # Total:
    # n: len(arr)
    # T(n) = 2T(n/2)+9 = 9 + 9logn
    # T(n) is O(n) = logn
    # -----------------------------------------------------------------------

    
    # inOrder function to print tree nodes in order
    #  @param self The object pointer
    #  @param root The originial input root
                                                    
    def inOrder(self,root):                                   # 1 step
        if root != None:                                      # 2 steps: if, !=
            Tree.inOrder(root.left)                           # T(n/2)
            print(root.data,end=" ")                          # 2 steps: root.data, print()
            Tree.inOrder(root.right)                          # T(n/2)
            
    # -----------------------------------------------------------------------
    # Total:
    # n: len(arr)
    # T(n) = 2T(n/2)+5= 5 + 5logn
    # T(n) is O(n) = logn
    # -----------------------------------------------------------------------


    ## inner tree node class
    class TreeNode:                                           # 1 step
        ##constructor for TreeNode
        #  @param self The object pointer
        #  @param data Data of the node
        def __init__(self, data):                             # 1 step
            self.data = data                                  # 1 step
            self.left = self.right = None                     # 2 steps: self.left=, self.right=
            
    # -----------------------------------------------------------------------
    # Total:
    # n: len(arr)
    # T(n) = 5 
    # T(n) is O(n) = 5
    # -----------------------------------------------------------------------
    
################################################################
        
## Subclass DecisionTree inherit from Tree and ClassifierAlgorithm
class DecisionTree(Tree,ClassifierAlgorithm):                # 1 step
    
    ##constructor for DecisionTree
    #  @param self The object pointer
    def __init__(self):                                      # 1 step
        
        #inheritance from the ClassifierAlgorithm constructor
        super().__init__()                                   # 1 step
        
        ##attribute type that is saved for default
        self.type = "continuous"                             # 1 step
    
    ##toString function to return the name of classifier
    #  @param self The object pointer
    def toString(self):                                      # 1 step
        return self.type + " Decision Tree"                  # 2 steps: self.type + "Decision Tree", return
        
    ##setType function to set k in the classifier type
    #  @param self The object pointer
    #  @param datatype Data type want to set
    def setType(self, datatype):                             # 1 step
        self.type = datatype                                 # 1 step
        print("type is saved")                               # 1 step
    
    
    ##testApplicable function to test whether the dataset is applicable for the classifier
    #  @param self The object pointer
    #  @param ds The data set pass in to test applicable
    #  @param l The label pass in to test applicable
    #  @return boolean whether the dataset is applicable
                                                             # **------ m: length of self.trainL ------**
    def testApplicable(self,ds,l):                           # 1 step
        try:                                                 # 1 step
            self.test(self.splitTrainTest(ds,l),self.type)   # 4m+13+2 steps: 4m+13 steps -- self.splitTrainTest(ds,l), 2 steps -- self.test()
            return True                                      # 1 step
        except:
            print(self.toString()+ " is not applicable for this dataset")
            return False

    ##entropy function to calculate entropy
    #  @param self The object pointer
    #  @param y Label value input
    #  @return entropy value
    def entropy(self,y):                                     # 1 step
       
        if y.size > 1:                                       # 3 steps: if, y.size,  >
    
            category = list(set(y))                          # 3 steps: set(y), list(), category()
        else:
    
            category = [y.item()]
            y = [y.item()]
    
        ent = 0                                              # 1 step
                                                             # **-- l: len(list(set(y))) --** #
        for label in category:                               # l steps
            p = len([label_ for label_ in y if label_ == label]) / len(y) 
                                                             # 6l steps: if, label_ == label, len([label_ for label_ in y]), len(y), /, p=
            ent += -p * math.log(p, 2)                       # 5l steps: math.log(), *, -p, +=, ent
    
        return ent                                           # 1 step
    # -----------------------------------------------------------------------
    # Total:
    # l: len(list(set(y)))
    # T(l) = 12l + 9
    # T(l) is O(l) = l
    # -----------------------------------------------------------------------

    
    
    ##Gini function to calculate Gini index for tree nodes as tree node
    #  @param self The object pointer
    #  @param y Label value input
    #  @return gini index value
    def Gini(self,y):                                        # 1 step
                                                             # **-- l: len(list(set(y))) --** #
            category = list(set(y))                          # 3 steps: set(y), list(), category = 
            gini = 1                                         # 1 step
        
            for label in category:                           # l steps
                p = len([label_ for label_ in y if label_ == label]) / len(y)
                                                             # 6l steps: if, label_ == label, len([label_ for label_ in y]), len(y), /, p=
                gini += -p * p                               # 4l steps: -p, *, +=, gini
        
            return gini                                      # 1 step
    # -----------------------------------------------------------------------
    # Total:
    # l: len(list(set(y)))
    # T(l) = 11l+ 6
    # T(l) is O(l) = l
    # -----------------------------------------------------------------------

    
    ##Gini_index_min function to get the minimal gini index for continuous value
    #  @param self The object pointer
    #  @param X The inputted training set
    #  @param y Label value input
    #  @param d The position of the decision attribute in X
    #  @return gini index value, thresh hold for the value in the position
    def Gini_index_min(self,X, y, d):                        # 1 step
     
        X = X.reshape(-1, len(X.T))                          # 4 steps: X.T, len(X.T), X.reshape(), X=
        X_attr = X[:, d]                                     # 1 step
        X_attr = list(set(X_attr))                           # 3 steps: set(X_attr), list(), X_attr=
        X_attr = sorted(X_attr)                              # 2 steps: sorted(X_attr), X_attr = 
        Gini_index = 1                                       # 1 step
        thre = 0                                             # 1 step

                                                             ## ** -- d: len(X_attr) -- ** ##
        for i in range(len(X_attr) - 1):                     # 3(d-1)+3 steps: 3 steps - len(X_attr), -1, range(); 3(d-1) steps: i < range, i++, i= 
            thre_temp = (X_attr[i] + X_attr[i + 1]) / 2      # 5(d-1) steps: X_attr[i], X_attr[i+1], +, /2, thre_temp=
            y_small_index = [i_arg for i_arg in range(len(X[:, d])) if X[i_arg, d] <= thre_temp]
                                                             # 4(d-1) steps: if, X[i_arg,d], <= thre_temp, y_small_index =
            y_big_index = [i_arg for i_arg in range(len(X[:, d])) if X[i_arg, d] > thre_temp]
                                                             # 4(d-1) steps: if, X[i_arg, d], > thre_temp, y_big_index=
            y_small = y[y_small_index]                       # 2(d-1) steps: y[y_small_index], y_samll=
            y_big = y[y_big_index]                           # 2(d-1) steps: y[y_big_index] ,  y_big =
                                                             
                                                             ## **-- l: len(list(set(y_small))) --** ##
            Gini_index_temp = (len(y_small) / len(y)) * self.Gini(y_small) + (len(y_big) / len(y)) * self.Gini(y_big)
                                                             # (10+2(11l+6))*(d-1) steps: 2(11l+6) steps- self.Gini(y_small), self.Gini(y_big); 10 steps- len(y_small), len(y), /, * , +, len(y_big), len(y), / ,  * , Gini_index_temp =
            if Gini_index > Gini_index_temp:                 # 2(d-1) steps: if, >
                Gini_index = Gini_index_temp                 # (d-1) steps
                thre = thre_temp                             # (d-1) steps
        return Gini_index, thre                              # 1 step
    
    # -----------------------------------------------------------------------
    # Total:
    # d: len(X_attr), l: len(list(set(y_small)))
    # T(d,l) = 46d+22ld-22l-29
    # T(d,l) is O(d,l) = ld 
    # -----------------------------------------------------------------------
    
    
    ## permutation function for discrete features
    #  @param self The object pointer
    #  @param s Input set for permutation
    #  @return result of permutation
                                                       ## ** -- s: len(s) -- ** ## 
    def permutation(self,s):                           # 1 step
        res = []                                       # 1 step
        c_len = len(s)                                 # 2 steps: len(s), c_len=
        for i in range(c_len):                         # 3s+1 steps: 1 step - range(); 3s steps - i<range, i++, i=
            for j in range(i + 1, c_len):              # (3(s-1)+1)*(3s+1) steps: 1 step - range(); 3(s-1) steps - i<range, i++, i=
                res.append((s[i], s[j]))               # 3*(3(s-1)+1)*(3s+1)steps: s[i],s[j],append()
        return res                                     # 1 step
    # -----------------------------------------------------------------------
    # Total:
    # s: len(s) 
    # T(s) = 36s^2-9s-2
    # T(s) is O(s) = s^2
    # -----------------------------------------------------------------------

    ##Gini_index_min_discrete function to get the minimal gini index for discrete value
    #  @param self The object pointer
    #  @param X The inputted training set
    #  @param y Label value input
    #  @param d The position of the decision attribute in X
    #  @return gini index value, attribute that split the node
    def Gini_index_min_discrete(self,X, y, d):        # 1 step
        X = X.reshape(-1, len(X.T))                   # 4 steps: X.T, len(X.T), X.reshape(), X=
        X_attr = X[:, d]                              # 1 step
        X_attr = list(set(X_attr))                    # 3 steps: set(X_attr), list(), X_attr=
        Gini_index = 1                                # 1 step
        attr = ()                                     # 1 step
        
                                                      ## ** -- d: len(X_attr) -- ** ##
        X_attr = self.permutation(X_attr)             # (36d^2-9d-2)+1 steps: 36d^2-9d-2 steps - self.permutation(); 1 step- X_attr=
    
        for X_pair in X_attr:                         # 3d steps: X_pair < len(X_attr), X_pair++, X_pair = 
            y_small_index = [i_arg for i_arg in range(len(X[:, d])) if X[i_arg, d] == X_pair[0]]
                                                      # 5d steps: if, X_pair[0], X[i_arg,d], == , y_small_index =
            y_big_index = [i_arg for i_arg in range(len(X[:, d])) if X[i_arg, d] == X_pair[1]]
                                                      # 5d steps: if, X[i_arg, d], X_pair[1], ==, y_big_index = 
            y_small = y[y_small_index]                # 2d steps: y[y_small_index], y_small=
            y_big = y[y_big_index]                    # 2d steps: y[y_big_index], y_big = 
            
                                                      ## **-- l: len(list(set(y_small))) --** ##
            Gini_index_temp = (len(y_small) / len(y)) * self.Gini(y_small) + (len(y_big) / len(y)) * self.Gini(y_big)
                                                      # (10+2(11l+6))*d steps: 2(11l+6) steps- self.Gini(y_small), self.Gini(y_big); 10 steps- len(y_small), len(y), /, * , +, len(y_big), len(y), / ,  * , Gini_index_temp =
            if Gini_index > Gini_index_temp:          # 2d steps: if, >
                Gini_index = Gini_index_temp          # d steps
                attr = X_pair                         # d steps
    
        return Gini_index, attr                       # 1 step
    
    
    

    ##attribute_based_on_Giniindex function to decide attribute based on gini index
    #  @param self The object pointer
    #  @param X The inputted training set
    #  @param y Label value input
    #  @param data_type The type of the tree
    #  @return gini index value, threshold or attribute, position of attribute in X
    def attribute_based_on_Giniindex(self,X, y, data_type):     # 1 step
       
        D = np.arange(len(X.T))                                 # 4 steps: X.T, len(), np.arrange(), D=
        Gini_Index_Min = 1                                      # 1 step
        Gini_index, thre = 0, 0                                 # 2 steps
        thre_ = 0                                               # 1 step
        d_ = 0                                                  # 1 step
        attr_ = ()                                              # 1 step
                                                                ## **-- x: len(inputted training set) --** ##
        for d in D:                                             # 3x steps: d < len(D), d++, d=
            if data_type == 'continuous':                       # 2x steps: if, == 
                                                                ## **-- d: len(X_attr), l: len(list(set(y_small))) --** ##
                Gini_index, thre = self.Gini_index_min(X, y, d) # ((46d+22ld-22l-29)+2)*x steps: 2 steps - Gini_index=, thre=; (46d+22ld-22l-29) steps: self.Gini_index_min() 
            elif data_type == 'discrete':
                Gini_index, attr = self.Gini_index_min_discrete(X, y, d)
            if Gini_Index_Min > Gini_index:                     # 2x steps: if, >
                Gini_Index_Min = Gini_index                     # x steps
                if data_type == 'continuous':                   # 2x steps: if, == 
                    thre_ = thre                                # x steps
                else:
                    attr_ = attr
                d_ = d                                          # x steps

        if data_type == 'continuous':                           # 2 steps: if, ==
            return Gini_Index_Min, thre_, d_                    # 1 step
        elif data_type == 'discrete':
            return Gini_Index_Min, attr_, d_
    # -----------------------------------------------------------------------
    # Total:
    # x: len(inputted training set), d: len(X_attr), l: len(list(set(y_small)))
    # T(x,d,l) = 46 dx + 22 ldx - 22 lx - 15 x + 14
    # T(x,d,l) is O(x,d,l) = ldx
    # -----------------------------------------------------------------------
    
    ##devide_group function to devide groups for continuous data
    #  @param self The object pointer
    #  @param X The inputted training set
    #  @param y Label value input
    #  @param thre Threshold splitting the group
    #  @param d The position of the decision attribute in X
    #  @return small group of X, small group of y, large group of X, large group of y
    def devide_group(self,X, y, thre, d):                        # 1 step
       
        
        x_small_index = [i_arg for i_arg in range(len(X[:, d])) if X[i_arg, d] <= thre]
                                                                 # 4 steps: if, X[i_arg,d], <= thre , x_small_index =
        x_big_index = [i_arg for i_arg in range(len(X[:, d])) if X[i_arg, d] > thre]
                                                                 # 4 steps: if, X[i_arg,d], > thre , x_big_index =
    
        X_small = X[x_small_index]                               # 2 steps: X[x_small_index], X_small=
        y_small = y[x_small_index]                               # 2 steps: y[x_small_index], y_small=
        X_big = X[x_big_index]                                   # 2 steps: X[x_big_index], X_big =
        y_big = y[x_big_index]                                   # 2 steps: y[x_big_index], y_big =
        return X_small, y_small, X_big, y_big                    # 1 step
    
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 18
    # T(n) is O(n) = 18
    # -----------------------------------------------------------------------
    

    ##devide_group_discrete function to devide groups for discrete data
    #  @param self The object pointer
    #  @param X The inputted training set
    #  @param y Label value input
    #  @param attr The attribute splitting the group
    #  @param d The position of the decision attribute in X
    #  @return small group of X, small group of y, large group of X, large group of y
    def devide_group_discrete(self,X, y, attr, d):             # 1 step
      
       
        x_small_index = [i_arg for i_arg in range(len(X[:, d])) if X[i_arg, d] == attr[0]]
                                                                # 5 steps: if, X[i_arg,d], attr[0], == , x_small_index =
        x_big_index = [i_arg for i_arg in range(len(X[:, d])) if X[i_arg, d] == attr[1]]
                                                                # 5 steps: if, X[i_arg,d], attr[1], == , x_big_index =
        X_small = X[x_small_index]                              # 2 steps: X[x_small_index], X_small=
        y_small = y[x_small_index]                              # 2 steps: y[x_small_index], y_small=
        X_big = X[x_big_index]                                  # 2 steps: X[x_big_index], X_big =
        y_big = y[x_big_index]                                  # 2 steps: y[x_big_index], y_big =
        return X_small, y_small, X_big, y_big                   # 1 step
    
    ##NtHt function to calculate NtHt for pruning
    #  @param self The object pointer
    #  @param y Label value input
    #  @return NtHt value
    def NtHt(self,y):                           # 1 step
                                                # **-- l: len(list(set(y))) --** #
        ent = self.entropy(y)                   # 12l+9+1 steps: 12l+9 steps - self.entropy(y) ; 1 step - ent = 
                                                ## **-- k: len(label_count.keys()), t: len(list(total_label)) -- ** ##
        data_probability =self.predictScore(y)  # 18k + 9t +20+1 steps: 18k + 9t +20 steps- self.predictScore(y); 1 step - data_probability = 
        #print('ent={},y_len={},all={},data_probability={}'.format(ent, len(y), ent * len(y),data_probability))
        return ent * len(y)                     # 3 steps: len(y), *, return
    
    # -----------------------------------------------------------------------
    # Total:
    # l: len(list(set(y))), k: len(label_count.keys()), t: len(list(total_label))
    # T(l,k,t) = 12l + 18k + 9t + 35
    # T(l,k,t) is O(l,k,t) = l+k+t
    # -----------------------------------------------------------------------
        

    ##maxlabel function to calculate max label when several labels is available
    #  @param self The object pointer
    #  @param y Label value input
    #  @return most common label
    def maxlabel(self,y):                       # 1 step
        label_ = Counter(y).most_common(1)      # 3 steps Counter(y), .most_common(1), label_=
        
        return label_[0][0]                     # 3 steps: label_[0], label_[0][0], return
    
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 7
    # T(n) is O(n) = 7
    # -----------------------------------------------------------------------
    
    ##predictScore function to predict score for the terminal node
    #  @param self The object pointer
    #  @param y Label value input
    #  @return prediction probability
    def predictScore(self,y):                        # 1 step
        total_label=np.unique(self.trainL_Num)       # 2 steps: np.unique(), total_label=
        predict_pro=dict()                           # 1 step
                                                           ## **-- k: len(label_count.keys()), t: len(list(total_label)) -- ** ##
        label,counts = np.unique(y,return_counts=True)     # 4 steps: return_counts=True, np.unique(), label=, counts=  
        label_count=dict(zip(label, counts))               # 3 steps: zip(), dict(), label_count = 
        if len(label_count.keys())>=2:                     # 4 steps: label_count.keys(), len(), if, >=
            for i in range(len(label_count.keys())):       # 3k+3 steps: 3k steps - i < range(), i++, i= ; 3 steps - label_count.keys(), len(), range() 
                Positive_location=np.where(label == list(label_count.keys())[i])  
                                                           # 6k steps: label_count.keys(),list(), list()[i], ==, np.where(), Positive_location=                                   
                predict_pro[list(label_count.keys())[i]]=(sum(counts[Positive_location].astype(np.float))/sum(counts)) 
                                                           # 9k steps: counts[Positive_location], counts[Positive_location].astype(), sum(counts[Positive_location].astype(np.float)), sum(counts), /, label_count.keys(), list(), list()[i],=
            for j in list(total_label):                    # 3t+1 steps: 3t steps - j< range(list()), j++, j= ; 1 step - list(total_label)
                if j not in list(label_count.keys()):      # 4t steps: label_count.keys(), list(), not in, if               
                    predict_pro[j]=0                       # 2t steps: [j], = 
        else:
            Positive_location=np.where(label== list(label_count.keys())[0])                                     
            predict_pro[list(label_count.keys())[0]]=(sum(counts[Positive_location].astype(np.float))/sum(counts)) 
            for i in list(total_label):
                if i!=list(label_count.keys())[0]:                    
                    predict_pro[i]=0
        return predict_pro                                 # 1 step
    
    # -----------------------------------------------------------------------
    # Total:
    # k: len(label_count.keys()), t: len(list(total_label))
    # T(k,t) = 18k + 9t +20
    # T(k,t) is O(k,t) = k+t
    # -----------------------------------------------------------------------

    # buildTree function to build decision tree
    #  @param self The object pointer
    #  @param X The inputted training set
    #  @param y Label value input
    #  @param data_type Data type of the tree, preset to continuous
    #  @param maxDepth Maximum depth of the tree, preset to 5
    #  @param currentDepth Current depth of the tree, preset to 0
    #  @param root The originial input root
    #  @return decisionnode object with left branches and right branches
    def buildTree(self,X, y, data_type='continuous',maxDepth=5,currentDepth = 0):
                                                               # 1 step
          
        Gain_max, thre, d = 1, 0, 0                            # 3 steps: Gain_max=, thre=, d=
        attr = ()                                              # 1 step
        X_small, y_small, X_big, y_big = [], [], [], []        # 4 steps: X_small=, y_small=, X_big=, y_big=

        ## when there is more than one y label
        if y.size> 1:                                          # 3 steps: y.size, if, >
           
            if data_type == 'continuous':                      # 2 steps: if, == 
                Gain_max, thre, d = self.attribute_based_on_Giniindex(X, y, data_type)
                                                               ## **-- x: len(inputted training set), d: len(X_attr), l: len(list(set(y_small))) --** ##
                                                               # 46 dx + 22 ldx - 22 lx - 15 x + 14 +3 steps: 46 dx + 22 ldx - 22 lx - 15 x + 14 steps - self.attribute_based_on_Giniindex(); 3 steps - Gain_max = , thre =, d = 
            elif data_type == 'discrete':
                Gain_max, attr, d = self.attribute_based_on_Giniindex(X, y, data_type)
            if Gain_max >= 0 and len(list(set(y))) > 1 and len(np.unique(X, axis=0))>1 and currentDepth < maxDepth:
                                                               # 13 steps: if, and, and, and, currentDepth < maxDepth, np.unique(X, axis=0), len(), >1, set(y), list(), len(), >1, Gain_max >= 0 
                if data_type == 'continuous':                  # 2 steps: if, == 
                    X_small, y_small, X_big, y_big = self.devide_group(X, y, thre, d)
                                                               # 18+4 steps: 18 steps - self.devide_group(X, y, thre, d); 4 steps - X_small = , y_small = , X_big = , y_big =
                elif data_type == 'discrete':
                    X_small, y_small, X_big, y_big = self.devide_group_discrete(X, y, attr, d)
                currentDepth+=1                                # 2 steps: +=1, currentDepth = 
                left_branch = self.buildTree(X_small, y_small,data_type,maxDepth,currentDepth=currentDepth)
                                                               # T(n/2)+1 steps, T(n/2) steps - self.buildTree(); 1 step - left_branch =
                right_branch = self.buildTree(X_big, y_big,data_type,maxDepth,currentDepth=currentDepth)
                                                               # T(n/2)+1 steps, T(n/2) steps - self.buildTree(); 1 step - right_branch =
                                                               # **-- l: len(list(set(y))), k: len(label_count.keys()), t: len(list(total_label)) --** ##
  
                nh = self.NtHt(y)                              # 12l + 18k + 9t + 35 + 1 steps: 12l + 18k + 9t + 35 steps - self.NtHt(y); 1 step  - nh =
                max_label = self.maxlabel(y)                   # 7 + 1 steps: 7 steps - self.maxlabel(y) ; 1 step - max_label=
                                                               ## **-- k: len(label_count.keys()), t: len(list(total_label)) -- ** ##
                prediction_prob=self.predictScore(y)           # 18k + 9t +20 +1 steps: 18k + 9t +20 steps - self.predictScore(y), 1 step - prediction_prob=
                return self.decisionnode(d=d, attr= attr, thre=thre, NH=nh, lb=left_branch, rb=right_branch, max_label=max_label,data_prob=prediction_prob)
                                                               # 10 + 1 steps: 10 steps - self.decisionnode(); 1 step - return
            else:
                nh = self.NtHt(y)
                max_label =self. maxlabel(y)
                prediction_prob=self.predictScore(y)
                return self.decisionnode(results=y[0], NH=nh, max_label=max_label,data_prob=prediction_prob)
        else:
            nh = self.NtHt(y)
            max_label = self.maxlabel(y)
            prediction_prob=self.predictScore(y)
            return self.decisionnode(results=y.item(), NH=nh, max_label=max_label,data_prob=prediction_prob)
    
    # -----------------------------------------------------------------------
    # Total:
    # x: len(inputted training set), d: len(X_attr), l: len(list(set(y_small))), k: len(label_count.keys()), t: len(list(total_label))
    # T(x,d,l,k,t) = 2T(n/2)+ 46dx + 22 ldx -22lx -15x + 12 l + 36k + 18t + 148 
    #              = logn + 46dx + 22 ldx -22lx -15x + 12 l + 36k + 18t + 148 
    # T(x,d,l,k,t) is O(x,d,l,k,t) = logn + ldx + k + t
    # -----------------------------------------------------------------------

    
    
    
    # classifyCon function to classify for continuous data
    #  @param self The object pointer
    #  @param observation The observation for classification
    #  @param tree The built tree for classification
    #  @return classification label and label probability
    ## classify for continuous data
    def classifyCon(self,observation, tree):     # 1 step
        if tree.results is not None:             
            return tree.max_label, tree.data_prob
        else:                                    # 1 step
            v = observation[tree.d]              # 3 steps: tree.d, observation[], v = 
    
            if v > tree.thre:                    # 3 steps: if, tree.thre, >
                branch = tree.rb                 # 2 steps: tree.rb, branch = 
            else:
                branch = tree.lb
    
            return self.classifyCon(observation, branch)  # T(n/2) + 1 steps: T(n/2)-self.classifyCon(); 1 step - return
    
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = T(n/2) + 10 = logn + 11 
    # T(n) is O(n) = logn
    # -----------------------------------------------------------------------
    
    

    # classifyDis function to classify for discrete data
    #  @param self The object pointer
    #  @param observation The observation for classification
    #  @param tree The built tree for classification
    #  @return classification label and label probability
    def classifyDis(self,observation, tree):    # 1 step
        if tree.results is not None:
            return tree.max_label,tree.data_prob
        else:                                   # 1 step
            v = observation[tree.d]             # 3 steps: tree.d, observation[], v = 
    
            if v == tree.attr[1]:               # 3 steps: tree.attr[1], if, ==
                branch = tree.rb                # 2 steps: tree.rb, branch = 
            else:
                branch = tree.lb
    
            return self.classifyDis(observation, branch) # T(n/2) + 1 steps: T(n/2)-self.classifyDis(); 1 step - return
    
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = T(n/2) + 10 = logn + 11 
    # T(n) is O(n) = logn
    # -----------------------------------------------------------------------
    
    
    
    # pruning function for pruning for overfitting
    #  @param self The object pointer
    #  @param tree The tree want to pruned
    #  @param alpha The pruning level, preset to 0.1
    def pruning(self,tree, alpha=0.1):          # 1 step
        if tree.lb.results is None:             
            self.pruning(tree.lb, alpha)
        if tree.rb.results is None:
            self.pruning(tree.rb, alpha)
        ## prune when right node and left node is not none
        if tree.lb.results is not None and tree.rb.results is not None:   # 4 steps: if, and, is not None, is not None
            before_pruning = tree.lb.NH + tree.rb.NH + 2 * alpha          # 7 steps: tree.lb, tree.lb.NH, tree.rb, tree.rb.NH, +, 2*alpha, before_pruning = 
            after_pruning = tree.NH + alpha                               # 3 steps: tree.NH, + , after_pruning =
            #print('before_pruning={},after_pruning={}'.format(before_pruning, after_pruning))
            if after_pruning <= before_pruning:                           # 2 steps: if, <=
                #print('pruning--{}:{}?'.format(tree.d, tree.thre))
                tree.lb, tree.rb = None, None                             # 2 steps: tree.lb=, tree.rb=
                tree.results = tree.max_label                             # 2 steps: tree.max_label, tree.results = 

    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 21
    # T(n) is O(n) = 21
    # -----------------------------------------------------------------------
    
   ##override test function to return test set
    #  @param self The object pointer
    #  @param testData The test dataset
    #  @param Datatype datatype for decision tree
    #  @param pruning Boolean for whether pruning the tree
    #  @return the prediction result of decision tree, the prediction probability, Label of probability
    def test(self,testData,Datatype=None,pruning =True):   # 1 step  
        
        super().test(testData)                             # 1 step
        
        if Datatype == None:                               # 2 steps: 'if' -- 1 step, '==' -- 1 step
            Datatype = self.type                           # 1 step, 1 space
                                                                                                # **------- m: length of self.trainL , n: length of testData -------**
        char_to_int = dict((c, i) for i, c in enumerate(self.trainL))                           # 3m+2 steps: i,c =, i,c < length of self.trainL, i,c ++, get (c,i) -- 3m steps, dict (c,i) -- 1 step, char_to_int = dict(c,i) -- 1 step    ; 5 space: i, c, (c,i), dict(),char_to_int
        self.trainL_Num= np.array([char_to_int[char] for char in self.trainL])                  # 3m+2 steps: char=, char< length of self.trainL, char ++, char_to_int[char] -- 3m steps, np.array() -- 1 step, self.tranL_Num = np.array() -- 1 step ; 3 space: char, np.array(), self.trainL_Num

        
        unique_train, counts_train=np.unique(self.trainL_Num, return_counts=True)               # m steps ; 3 space: unique_train, counts_train, np.unique()
        train_label_count=dict(zip(unique_train, counts_train))                                 # m+2 steps: zip -- m steps, dict() -- 1 step, train_label_count=dict() -- 1 step  ; 2 space: dic(), train_label_count       
        self.prediction_score= [[] for _ in unique_train]                                       # 3m+1 steps: _ =, _< length of unique_train, _++, [] -- 3m steps, self.prediction_score = [] -- 1 step ; 3 space: _, [], self.prediction_score
            
        int_to_char = {v: k for k, v in char_to_int.items()}                                    # 3m+2 steps: k,v = ,k,v < length(char_to_int.items(), k,v++, get v:k -- 3m steps, items() -- 1 step, int_to_char= v:k -- 1 step ; 4 space: k, v, {v:k}, int_to_char                              
        ## for continuous data                                                                  #------* Worst case here is when the data is continuous (since it has one more step for 'astype()') *-------#
        if Datatype=="continuous":                                                              # 2 steps: 'if' -- 1 step, '==' -- 1 step  
            self.tree = self.buildTree(self.trainDS.astype(np.float),self.trainL_Num,Datatype)  # 3 steps: self.trainDS.astype() -- 1 step, self.buildTree() -- 1 step, self.tree = self.buildTree() -- 1 step   ; 2 space: self.trainDS.astype(np.float), self.tree
            #print("finish building tree")
            if pruning == True:                                                                 # 2 steps: 'if' -- 1 step, '==' -- 1 step 
                self.pruning(self.tree)                                                         # 1 step: self.pruning()  
            self.prediction= []                                                                 # 1 step  ; 1 space
            for i in testData:                                                                  # n*3 steps: i =, i < length(testData), i++ -- 3 steps ; 1 space: i
                prediction_label,prediction_prob=self.classifyCon(i.astype(np.float),self.tree) # n*3 steps: i.astype() -- 1 step, self.classifyCon() -- 1 step, prediction_label,prediction_prob = self.classifyCon() -- 1 step   ; 3 space: prediction_label, prediction_prob, i.astype(np.float) 
                self.prediction.append(prediction_label)                                        # n*1 steps: append() -- 1 step
                for j in range(len(train_label_count.keys())):                                  # n*m*3+3 steps: keys() -- 1 step, len() -- 1 step, range() -- 1 step, j =, j<range(len()), j ++ --3 steps  ; 4 space: j, keys(), len(),range()                  
                    self.prediction_score[j].append(prediction_prob.get(list(int_to_char)[j]))  # n*m*3 steps: list() -- 1 step, prediction_prob.get() -- 1 step, append() -- 1 step 

        ## for discrete data
        else:
            self.tree = self.buildTree(self.trainDS,self.trainL_Num,Datatype)
            #print("finish building tree")
            if pruning == True:
                self.pruning(self.tree)
            self.prediction= []  
    
            for i in testData:
                prediction_label,prediction_prob=self.classifyDis(i,self.tree)
                self.prediction.append(prediction_label)
                for j in range(len(train_label_count.keys())):                                  
                    self.prediction_score[j].append(prediction_prob.get(list(int_to_char)[j])) 
        ## turn label encoder to real label
        self.prediction= [int_to_char[char] for char in self.prediction]                      # 3m+1 steps: char =, char<length(self.prediction), char++, get int_to_char[char] -- 3m steps, self.prediction = int_to_char -- 1 step 
        ## return predication
        return self.prediction,self.prediction_score, list(char_to_int)                       # 2 steps: list() -- 1 step, return -- 1 step  ; 1 space: list() 


    # -----------------------------------------------------------------------
    # Total:
    # m: length of self.trainL , n: length of testData
    # T(n,m) = 6mn + 18m + 7n + 29
    # T(n,m) is O(n,m) = mn+ m + n
    # S(n,m)= 30
    # S(n,m) is O(n,m) = 30
    # -----------------------------------------------------------------------

      
################################################################
    ## inner class decision tree node， return all the attributes of trees
    class decisionnode:             
        
        ##constructor for decisionnode
        #  @param self The object pointer
        #  @param d Decided attribute saved in the node
        #  @param attr Attribute saved in the node
        #  @param thre Threshold stored in the node
        #  @param results The final results of the node
        #  @param NH NH value stored in the tree
        #  @param lb The right branch of the node
        #  @param rb The right branch of the node
        #  @param max_label The most occured label in each node
        #  @param data_prob Probility for each label stored in the node
        def __init__(self, d=None, attr = None,thre=None, results=None, NH=None, lb=None, rb=None, max_label=None,data_prob=None):
                                     # 1 step
            self.d = d               # 1 step
            self.thre = thre         # 1 step
            self.results = results   # 1 step
            self.NH = NH             # 1 step
            self.lb = lb             # 1 step
            self.rb = rb             # 1 step
            self.max_label = max_label  # 1 step
            self.attr = attr            # 1 step
            self.data_prob=data_prob    # 1 step
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 10
    # T(n) is O(n) = 10
    # -----------------------------------------------------------------------

    
##################################################        
### ------------------------------------------ ###
## Total for DecisionTree:
## x: len(inputted training set), d: len(X_attr), l: len(list(set(y_small))),
## k: len(label_count.keys()), t: len(list(total_label)), m: length of self.trainL, n: length of testData
    
## T(x,d,l,k,t,m,n) = 3logn + 46dx + 22 ldx -22lx -15x + 12 l + 36k + 18t + 6mn + 22m + 7n + 248
## T(x,d,l,k,t,m,n) is O(x,d,l,k,t,m,n) = logn+ldx+k+t+mn
### ------------------------------------------ ###  
##################################################  
            
################################################################

## Class Experiment
class Experiment:  
    
    ##constructor for Experiment
    #  @param self The object pointer
    #  @param dataset The dataset that take classifier, numpy type, without header
    #  @param label The label for dataset, numpy type, without header
    #  @param classifierList The classifier used for experiment
    def __init__(self, dataset, label,classifierList):
        ## attribute ds to save the dataset
        self.ds = dataset
        ## attribute l to save the label
        self.l = label
        ## attribute classifier which is a list to save availiable classifier
        self.classifier = []     
                    
        ## attribute to store label type
        self.label_type=np.unique(self.l) 


        # try each classifier, now only knn and decision tree
        for c in classifierList:
            if c.testApplicable(self.ds,self.l):
                self.classifier.append(c)



    ##runCrossVal function that run cross validation
    #  @param self The object pointer
    #  @param k Number of folder in cross validation
    def runCrossVal(self, k):
        ##attribute dataset_split to save the splitted data
        self.dataset_split = list()
        ##attribute dataset_split to save the splitted data
        self.prediction = []
        ##attribute dataset_split to save the splitted data
        self.true=[]
        ## prediction score for positive class
        self.prediction_score=[[] for _ in np.unique(self.l)] 
        # Seperate to k folders
        dataset_index = list(list(range(len(self.l))))
        fold_size = int(len(self.l) / k)
        for i in range(k):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_index))
                fold.append(dataset_index.pop(index))
            self.dataset_split.append(fold)

        # Make prediction according to folders
        for c in range(len(self.classifier)):
            prediction = []
            ps = [[] for _ in np.unique(self.l)] 
            true = []

            for j in range(k):
                trainds = np.delete(self.ds,self.dataset_split[j],0)
                trainl = np.delete(self.l,self.dataset_split[j],0)
                testds = self.ds[self.dataset_split[j]]
                testl = self.l[self.dataset_split[j]]
                self.classifier[c].train(trainds,trainl)
                test1,test2,test3=self.classifier[c].test(testds)
                prediction.extend(test1)
                for i in range(len(self.label_type)):
                    ps[i].extend(test2[test3.index(self.label_type[i])])
                    
                true.extend(testl.tolist())
            for i in range(len(np.unique(self.l))):
                self.prediction_score[i].append(ps[i])
            
 
            self.prediction.append(prediction)
            self.true.append(true)
            
    
     
    ##score function that return score value
    #  @param self The object pointer
    def score(self):                                                     # 1 step
                                                                         # n: len(self.classifier)
        print ('%-30s%-20s' % ("Classifier Name", "Accruacy Rate"))      # 1 step 
        for c in range(len(self.classifier)):                            # 2n+2 steps, 4 space
            truecount = 0                                                # n step, 1 space
            count= len(self.prediction[c])                               # 2n steps, 1 space
                                                                         # m: len(self.prediction[c])
            for i in range(len(self.prediction[c])):                     # 2m+3 steps, 5 space
                if self.prediction[c][i] == self.true[c][i]:             # 4m steps, 4 space
                    truecount+= 1                                        # 2m steps
            accuracy = 100*truecount/count                               # 3n steps, 1 space
                
            print ('%-30s%-20s' % (self.classifier[c].toString(), str(accuracy)+" %")) # 3n steps, 2 space
     
    #--------------------------------------------------------------------------------------------------
    # Total:
    # T(n,m) = (8m+3)*n+6n+2n+2+3n+1
    # T(n,m) is O(n,m) = mn
    # S(n,m)= 16
    # S(n,m) is O(n,m) = 16
    # -----------------------------------------------------------------------  
        
    ##confusionMatrix function that build and return the confusion matrix
    #  @param self The object pointer
    def confusionMatrix(self):                                                 # 1 step
                                                                               # n: len(self.classifier)
        for c in range(len(self.classifier)):                                  # 2n+2 steps,4 space
            # caluclate confusion matrix by first get unique value of label    
                                                                               # m: len(set(self.true[c]))
            unique = sorted(set(self.true[c]))                                 # n*mlog(m)+n+n+n steps, 1 space
                        
            # create empty matrix                                              
            matrix = [[0 for _ in unique] for _ in unique]                     # m^2*n steps, 2 space
            # create dictionary
            imap   = {key: i for i, key in enumerate(unique)}                  # 5m*n steps, 4 space
            # Generate Confusion Matrix
            for p, a in zip(self.prediction[c], self.true[c]):                 # 7mn steps, 7 space
                matrix[imap[p]][imap[a]] += 1                                  # 5mn steps
                
            # output formatted confustion matrix:
            print ('\nConfusion matrix for ' + self.classifier[c].toString()+', row for predicted label, column for true label')
                                                                               # 3n steps, 1 space
            for row in range(len(unique)+1):                                   # (2m+3)*n steps, 3 space
                output=''                                                      # (m+1)*n steps, 1 space
                if row == 0:                                                   # 2(m+1)*n steps, 2 space
                    output+= "{:<15}".format(" ")                               # 3(m+1)*n steps, 2 space
                    for l in unique:                                           # 2m(m+1)*n steps, 2 space
                        output+= "{:<15}".format(l)                             # 3m(m+1)*n steps, 2 space
                else:
                    output+="{:<15}".format(unique[row-1])
                    for i in range(len(matrix)):
                        output+= "{:<15}".format(matrix[row-1][i])
                print(output)                                                  # (m+1)*n steps

    #--------------------------------------------------------------------------------------------------
    # Total:
    # T(n,m) = 2n+2 + n*mlog(m)+n+n+n  + m^2*n + 5m*n + 7mn +5mn + 3n +  (2m+3)*n +  (m+1)*n+ 2(m+1)*n + 3(m+1)*n + 2m(m+1)*n  + 3m(m+1)*n
    # T(n,m) is O(n,m) = m^2*n 
    # S(n,m)= 31
    # S(n,m) is O(n,m) = 31
    # -------------------------------------------------------------------------                 
        
    ##drawConfusionMatrix function that draw one confusion matrix for one specific classifier
    #  @param self The object pointer
    #  @param classifierInd The classifier index want to draw
    def drawConfusionMatrix(self,classifierInd):
        # caluclate confusion matrix by first get unique value of label
        unique = sorted(set(self.true[classifierInd]))
        # create empty matrix
        matrix = [[0 for _ in unique] for _ in unique]
        # create dictionary
        imap   = {key: i for i, key in enumerate(unique)}
        # Generate Confusion Matrix
        for p, a in zip(self.prediction[classifierInd], self.true[classifierInd]):
            matrix[imap[p]][imap[a]] += 1

        
        z = matrix
        
        x = unique
        y = unique
        
        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in z]
        time.sleep(5)
        # set up figure 
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
        
        # add title
        fig.update_layout(title_text='<i><b>Confusion matrix for ' + self.classifier[classifierInd].toString()+'</b></i>',
                          xaxis = dict(title='predicted'),
                          yaxis = dict(title='real')
                         )
        
        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        
        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.35,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        
        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))
        
        # add colorbar
        fig['data'][0]['showscale'] = True
        py.plot(fig)
        

    
    ##TPR_NPR_Calculation function that calculates true positive rate and false positive rate
    #  @param self The object pointer
    #  @param c It presents number of alogorithms used in experiment class
    #  @param i It presents location of prediction_score for more than two classes dataset when using one class as positive class and all other classes as negative class
    #  @return FPR and TPR . False positive rate and true positive rate
                                                                               # ------------------------------------------------------------------------------------------------------- # 
    def TPR_FPR_Calculation(self,c=None,i=None):                               # 1 step
        #c=0
        #K=2
        #self.runCrossVal(K,i)                                                 # **-- n: len(self.true[c]) --** 
        #print(i)
        true_label=np.array(self.true[c])                                      # 3 steps: self.true[c], np.array(), true_label =        
        unique, counts=np.unique(true_label, return_counts=True)               # 4 steps: return_counts = True, np.unique(), unique =, counts =
        label_count=dict(zip(unique, counts))                                  # 3 steps: zip(), dict(), label_count=
        prediction_score=np.array(self.prediction_score[i][c])                 # 4 steps: self.prediction_score[i], self.prediction_score[i][c], np.array(), prediction_score = 
        index=np.argsort(-1*prediction_score.astype(np.float))                 # 4 steps: prediction_score.astype(np.float), -1*, np.argsort(), index =
    
        true_label=true_label[index]                                           # 2 steps: true_label[index], true_label =
        prediction_score=prediction_score[index]                               # 2 steps: prediction_score[index], prediction_score =
              
        
        P=float(label_count[list(label_count.keys())[i]])                      # 6 steps: label_count.keys(), list(), list()[i], label_count[], float(), P =
        N=sum(label_count.values())-P                                          # 4 steps: label_count.values(), sum(), - P, N
        j=0                                                                    # 1 step
        FP=0                                                                   # 1 step
        TP=0                                                                   # 1 step
        FPR=[]                                                                 # 1 step
        TPR=[]                                                                 # 1 step
        prev_score=-1                                                          # 1 step
                                                                               ## ** -- The worst case here is when j  <= (len(true_label)-1) -- ** ##
        while j  <= (len(true_label)-1):                                       # 2(n-1)+2 steps: 2 steps -- len(), -1; 2(n-1) steps -- <=, while
                                                                               ## ** -- The worst case here is when prediction_score[j] != prev_score or j==(len(true_label)-1) -- ** ##
            if  prediction_score[j] != prev_score or j==(len(true_label)-1):   # 7(n-1) steps: len(), len()-1, j == , or, prediction_score[j], !=, prev_score
                FPR.append(FP/N)                                               # 2(n-1) steps: FP/N, FPR.append()
                TPR.append(TP/P)                                               # 2(n-1) steps: TP/P, TPR.append()
                prev_score=prediction_score[j]                                 # 2(n-1) steps: prediction_score[j], prev_score = 
            if  true_label[j] == list(label_count.keys())[i]:
                TP+=1
            else:
                FP+=1
            j=j+1    
        FPR.append(FP/N)                                           
        TPR.append(TP/P)                                                       # ------------------------------------------------------------------------------------------------------- # 
        
        
    # -----------------------------------------------------------------------
    # Total:
    # n: len(self.true[c])
    # T(n) = 15n + 26
    # T(n) is O(n) = n
    # -----------------------------------------------------------------------
            
        """    
        else:
            true_label=np.array(self.true)
            unique, counts=np.unique(true_label, return_counts=True)
            label_count=dict(zip(unique, counts))
            print(self.prediction_score[i])
            prediction_score=np.array(self.prediction_score[i]) 


            index=np.argsort(-1*prediction_score.astype(np.float))
        
            true_label=true_label[index]
            prediction_score=prediction_score[index]    
                  
            
            P=float(label_count[self.label_type[i]])
            N=sum(label_count.values())-P
            j=0
            FP=0
            TP=0
            FPR=[]
            TPR=[]
            prev_score=-1
            while j  <= (len(true_label)-1):
                if  prediction_score[j] != prev_score or j==(len(true_label)-1):
                    FPR.append(FP/N)
                    TPR.append(TP/P)
                    prev_score=prediction_score[j]
                if  true_label[j] == self.label_type[i]:
                    TP+=1
                else:
                    FP+=1
                j=j+1    
            FPR.append(FP/N)
            TPR.append(TP/P)
        """
        return  FPR,TPR
    
    ##Roc curve function that draw roc curves for different classifers
    #  @param self The object pointer
                                                                                # ------------------------------------------------------------------------------------------------------- # 
    def ROC_Curve(self):                                      # 1 step
                                                                                # **-- m: length of self.classifier[0].trainL , n: length of testData, k: len(self.classifier) -- ** 
        for j in range(len(self.classifier)):                                   # k*3+2 steps: j=, j< range(), j++ -- 3 steps, len() -- 1 step, range() -- 1 step ; 3 space: j, range(), len() 
            unique_train, counts_train=np.unique(self.classifier[0].trainL, return_counts=True)
                                                                                # k*(m+3) steps: get unique_train, counts_train -- m steps, self.classifier[0] -- 1 step, self.classifier[0].trainL -- 1 step, np.unique() -- 1 step; 3 space: unique_train, counts_train, np.unique()
            train_label_count=dict(zip(unique_train, counts_train))             # k*(m+2) steps: zip -- m steps, dict() -- 1 step, train_label_count=dict() -- 1 step  ; 2 space: dic(), train_label_count
            for i in range(len(train_label_count.keys())):                      # k*(m*3+3) steps: i=, i < range(), i++ -- 3 steps, keys() -- 1 step, len() -- 1 step, range -- 1 step ; 4 space: i, keys(), len(),range() 
                FPR,TPR=self.TPR_FPR_Calculation(j,i)                         # k*m*2 steps: self.TPR_NPR_Calculation() -- 1 step, FPR,TPR =self.TPR_NPR_Calculation() -- 1 step  ; 2 space: FPR, TPR    
                la=('ROC curve for %s when Positive class is %s' % (self.classifier[j].toString(),self.label_type[i],))
                                                                                # k*m*1 steps ; 1 space: la
                plt.plot(FPR,TPR,label=la)                                      # k*m*1 steps 


                    
        plt.legend(bbox_to_anchor=(1.1, 1.05))                                  # 1 step
        plt.title('ROC curve' )                                                 # 1 step
        plt.show()                                                              # 1 step
                                                                                # ------------------------------------------------------------------------------------------------------- # 
        
    # -----------------------------------------------------------------------
    # Total:
    # m: length of self.classifier[0].trainL , n: length of testData, k: len(self.classifier) 
    # T(m,k) = 9 mk + 11k + 6
    # T(m,k) is O(m,k) = mk
    # S(m,k)= 15
    # S(m,k) is O(m,k) = 15
    # -----------------------------------------------------------------------
  
      
                  
##################################################        
### ------------------------------------------ ###
## Total for ROC:
## m: length of self.classifier[0].trainL , n: length of testData, k: len(self.classifier) 
## T(m,n,k) = 9 mk +11k + 15n + 32
## T(m,n,k) is O(m,n,k) = mk+n
### ------------------------------------------ ###  
##################################################  
      

################################################################
    
# Class HeterogenousDataSets     
class HeterogenousDataSets():
    ##constructor for DataSet
    #  @param self The object pointer
    #  @param list_of_dataets The list of datastes that will be used to build HeterogeneousDataSets
    #  @param ds A list that contains csv file names
    def __init__(self, list_of_dataets):
        
        #set the attributes in the constructor
        self.lis_datasets = list_of_dataets
        self.ds = []
        
        # use function load to check whether the file is csv file so the constituent data sets could read in the datasets successfully
        for i in range(len(self.lis_datasets)):
            self.ds.append(self.__load(self.lis_datasets[i]))
        
            
    
    
    ##loadDataset function to read data from specific files
    #  @param self The object pointer
    #  @param hetero A list contains several DataSets according to specific types of them
    #  @param typ A list of the types of datasets
    #  @param ds A list that contains csv file names
    def loadDataset(self, type_of_datasets):
        # first create an empty list to contain the Datasets according to specific types of them
        self.hetero = []
        # get and save the types of datasets
        self.typ = type_of_datasets
        # For each dataset in the list,
        for i in range(len(self.ds)):
            # get the type of the dataset
            print("The type for dataset "+str(i+1)+ " is :", self.typ[i])

           
            # if the type of dataset is TimeSeriesDataset, apply the class TimeSeriesDataSet() on the dataset 
            if self.typ[i] == 'TimeSeriesDataset':
                self.hetero.append(TimeSeriesDataSet(self.ds[i]))
            
            # if the type of dataset is TextDataSet, apply the class TextDataSet() on the dataset
            elif self.typ[i] == 'TextDataSet':
                self.hetero.append(TextDataSet(self.ds[i]))
            
            # if the type of dataset is QuantDataSet, apply the class QuantDataSet() on the dataset
            elif self.typ[i] == 'QuanDataSet':
                self.hetero.append(QuantDataSet(self.ds[i]))
            
            # if the type of dataset is QualDataSet, apply the class QualDataSet() on the dataset
            elif self.typ[i] == 'QualDataSet':
                self.hetero.append(QualDataSet(self.ds[i]))
            
            # if the type of dataset is TransactionDataSet, apply the class TransactionDataSet() on the dataset
            elif self.typ[i] == 'TransactionDataSet':
                self.hetero.append(TransactionDataSet(self.ds[i]))
            
            # if the type does not belong to TimeSeriesDataset, TextDataSet, QuantDataSet, QualDataSet, and TransactionDataSet, raise an error
            else:
                raise AttributeError("The dataset must be in one of the types among TimeSeriesDataset, TextDataSet, QuantDataSet, QualDataSet, and TransactionDataSet.")
               

        
      
        
    ##savetype function to save the type of datasets
    #  @param self The object pointer
    #  @param typ A list of the types of datasets
    #  @return a list containing the types of datasets 
    def types(self):
        return self.typ
    
    
    ##datasets function to save the names of datasets
    #  @param self The object pointer
    #  @param ds A list of the names of datasets
    #  @return a list containing the names of datasets 
    def datasets(self):
        return self.ds
        
        
        
        
    ##load function to detect if the file is csv
    #  @param self The object pointer
    #  @param filename The file name dataset read from
    #  @return filename The file name dataset read from
    def __load(self, filename):

        #check if the dataset is csv 
        if (filename.endswith(".csv")):
            #if the dataset is csv, allow further process on constituent data sets using the filename inputted
            try:
                #try return the filename 
                return filename
            # if the dataset is not csv, return error
            except ImportError:
                print("The csv file does not exist")
        else:
            # print error message if the file name enter is not csv file
            print("Enter a csv file to load data set")
    
        #else if (filename.endswith()):
            #if it is not csv file, an example of read pds file is given and will be modify later

            #reurn tabula.read_pds(filename, encoding='utf-8', spreadsheet=True, pages='1-6041')
        

                
    ##clean function to call the clean methods from each of the individual datasets
    #  @param self The object pointer    
    #  @param hetero A list contains several DataSets according to specific types of them
    def clean(self):
        # For each dataset with its specific type,
        for j in range(len(self.hetero)):
            # print out the result of cleaning
            print(self.hetero[j].clean())
    
    
    
    ##explore function to call the explore methods from each of the individual datasets
    #  @param self The object pointer    
    #  @param hetero A list contains several DataSets according to specific types of them
    def explore(self):
        # For each dataset with its specific type,
        for j in range(len(self.hetero)):
            # print out the result of visualization 
            print(self.hetero[j].explore())
     
        
    ##select function to select one of the constituent data sets   
    #  @param self The object pointer 
    #  @param ds A list that contains csv file names 
    #  @param typ A list of the types of datasets
    #  @return The data set selected
    def select(self,index_datasets):
        try:
            print('The dataset you selected is:', self.ds[index_datasets])
            print('The type of the dataset you selected is:', self.typ[index_datasets])
            
            return(self.ds[index_datasets])
            
        except:
            print('Please put in the index of the selected dataset.')
        
        


    ##summary function to print out basic information about the lists of datasets
    #  @param self The object pointer    
    #  @param ds A list that contains csv file names 
    #  @param typ A list of the types of datasets
    def summary(self):
        # print the numbers of datasets
        print('There are totally', len(self.ds), 'datasets in the list\n')
        # print the names of datasets
        print('The names of the datasets are:', self.ds )
        # print the types of datasets
        print('The types of the datasets are:', self.typ)
        # For each dataset with its specific type,
        print('')
        print('')
        for j in range(len(self.hetero)):
            print('')
            # print out the result of visualization 
            print("The summary for",self.ds[j],'(',self.typ[j],') is:') 
            print(self.hetero[j].summary())


################################################################

## Subclass lshkNNClassifier inherit from ClassifierAlgorithm
class lshkNNClassifier(ClassifierAlgorithm):  
    
    ##constructor for simplekNNClassifier
    #  @param self The object pointer
    #  @param k Number of neighbors used to do classification
    #  @param l Length of binary code    
    def __init__(self,k,l):
        
        #inheritance from the ClassifierAlgorithm constructor
        super().__init__()
        
        ##attribute k that is saved for default
        self.k = k
        self.l=l


        
    ##calculate hamming distance for two given binary codes 
    #  @param self The object pointer
    #  @param binary_code1 a binary code which has been transfered from a normal feature vector  
    #  @param binary_code2 a binary code which has been transfered from a normal feature vector       
    def hamming_distance(self,binary_code1,binary_code2):
        
        dist=0
        for n in range(len(binary_code1)):
            if binary_code1[n] != binary_code2[n]:
                dist+=1
        return dist
        
               

    ##override train function to save train set
    #  @param self The object pointer
    #  @param trainData The training dataset
    #  @param labl Training label
    def train(self, trainData, label):
        self.trainDS = trainData        
        ##attribute trainL for saving training Label
        self.trainL = label   
        self.row,self.col=self.trainDS.shape
        self.train_hash=self.random_projection_hash(self.trainDS)  
        return self.train_hash                         


    ##predict label for a given binary code
    #  @param self The object pointer
    #  @param test_row the binary code which has been transfered from a normal feature vector             
    def random_projection_hash(self, dataset_without_label):
        self.projections=np.random.randn(self.col,self.l)        
        self.hash_table = list()   
        arr_dataset=np.array(dataset_without_label).astype(np.float)
        for row in arr_dataset:
            binary_code= (np.dot(row,self.projections) >= 0).astype(int)
            self.hash_table.append(''.join(binary_code.astype('str')))            
        return self.hash_table
    

    ##prediction_classification predict label for a given binary code according to k-nearest neighbor
    #  @param self The object pointer
    #  @param test_row the binary code which has been transfered from a normal feature vector     
    def find_neighbors(self,test_row):
        distance= list()
        i=0
        for train_row in self.train_hash:           
            dist=self.hamming_distance(test_row,train_row)
            distance.append((self.trainL[i],dist))
            i+=1
        distance.sort(key=lambda tup: tup[1])
        neighbor= list()
        for i in range(self.k):
            neighbor.append(distance[i][0])        
        return neighbor





        
    ##override test function to return test set
    #  @param self The object pointer
    #  @param testData TestDataset without label column
    #  @param label True labels for test dataset
    def test(self,testData):
        
        unique_train, counts_train=np.unique(self.trainL, return_counts=True)
        train_label_count=dict(zip(unique_train, counts_train))
        
        self.prediction_score= [[] for _ in unique_train] 
        self.prediction = []
        testData_hash=self.random_projection_hash(testData)
        for row in testData_hash:
            neighbor=self.find_neighbors(row)
            lab,count = np.unique(neighbor,return_counts=True)   
                                                                     
            for i in range(len(train_label_count.keys())):
                Positive_location=np.where(lab == list(train_label_count.keys())[i])                                     
                self.prediction_score[i].append(sum(count[Positive_location].astype(np.float))/sum(count))  
                
            predicted_class=lab[np.argmax(count)]            
            self.prediction.append(predicted_class)  
  
        return self.prediction,self.prediction_score,list(train_label_count)  
    
    ##testApplicable function to test whether the dataset is applicable for the classifier
    #  @param self The object pointer
    #  @param ds The data set pass in to test applicable
    #  @param l The label pass in to test applicable
    #  @return boolean whether the dataset is applicable
    def testApplicable(self,ds,l):
        try:
            self.test(self.splitTrainTest(ds,l))
            return True
        except:
            print(self.toString()+ " is not applicable for this dataset")
            return False    
    
    ##toString function to return the name of classifier
    #  @param self The object pointer
    def toString(self):
        return "lshKNNClassifier with k as " + str(self.k)
   
        

    


################################################################

## Subclass kdTreeKNNClassifier inherit from ClassifierAlgorithm
class kdTreeKNNClassifier(ClassifierAlgorithm):  
    
    ##constructor for kdTreeKNNClassifier
    #  @param self The object pointer
    def __init__(self):            # 1 step
        ##inheritance from the ClassifierAlgorithm constructor
        super().__init__()         # 1 step
        
        ##attribute k that is saved for default
        self.k = 3                 # 1 step
                  
    
    
    class KdTree:                     # 1 step
        def __init__(self, n_dim):    # 1 step
            """
            :param n_dim:  the dimension of this kd_tree
            """
            self.n_dim = n_dim   # the dimension of this tree   # 1 step
            self.root = None          # 1 step
            self.size = 0             # 1 step
    
        def distance(self, a, b):     # 1 step
            """
            :param a: type iterable
            :param b: type iterable
            :return: type double, return the Euclidean distance between a and b
            """
            s = 0.0                   # 1 step
            for x, y in zip(a, b):    # 3 steps: zip(a,b), for, in 
                d = x - y             # 2 steps: -, = 
                s += d * d            # 2 steps: +=, *
            return math.sqrt(s)       # 2 steps: math.sqrt(),return
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 11
    # T(n) is O(n) = 11
    # -----------------------------------------------------------------------  
        

        def createTree(self, data,  current_node,axis=0):   # 1 step
            """
            function to create a kd-tree, recursion
            :type data: array-like, samples used to construct a kd-tree or sub_kd-tree, the last column is label
            :type axis: int, between 0 and n_dim, dimension used to split data
            :type current_node: Node, the current 'root' node
            :return: None
            """
            if self.size == 0:                 # 2 steps: if, ==
                self.root = current_node       # 1 step
            self.size += 1                     # 1 step
            if data.shape[0] == 1:   # if no more than one sample, then stop iterating # 3 steps: if, data.shape[0], == 
                current_node.point = data[0, :]    # 3 steps: data[0,:], current_node.point, = 
                current_node.axis = axis           # 2 steps: current_node.axis, = 
                return                             # 1 step
            """
            step1: split the points with the median on this axis
            To find the median on target axis, we've got two ways:
            A. simply sort on target axis each time
            B. presort on each axis? but I haven't solved this at present
            """
            temp = data[data[:, axis].argsort()]    # 4 steps: data[:,axis], data[:,axis].argsort(), data[], temp=
            med = int(len(temp)/2) if len(temp) % 2 == 0 else int((len(temp)-1)/2)   # get the median of this axis
            # find the 'first' med, this means that "<" goto left child, ">" goto right child
                                                    # 9 steps: if, len(), % , ==, int(),len(), /, 2, med =
    
            while med > 0 and temp[med,axis] == temp[med-1, axis]:  # 7 steps: while, >0, and, temp[med,axis], == , temp[med-1,axis], med-1
                med -= 1                            # 1 step
            current_node.axis = axis                # 2 steps: current_node.axis, = 
            current_node.point = temp[med]          # 2 steps: current_node.point, = 
            tt = temp[med]                          # 2 steps: temp[med], tt=
            axis = (axis + 1) % self.n_dim          # 4 steps: self.n_dim, %, axis+1, axis = 
            if temp[:med, :].shape[0] >= 1:         # 4 steps: if, temp[], temp[].shape[0], >=
                tt = temp[:med, :]                  # 2 steps: temp[], tt=
                current_node.left = self.Node()     # 2+5 steps: 2 steps- current_node.left, = ; 5 steps -self.Node()
                self.createTree(temp[:med, :], current_node.left, axis)  # T(n/2)+2 steps: 2 steps-current_node.left, temp[], T(n/2) steps - self.createTree()
            if temp[(med+1):, :].shape[0] >= 1:     # 4 steps: if, temp[], temp[].shape[0], >=
                tt = temp[(med+1):, :]              # 2 steps: temp[], tt=
                current_node.right = self.Node()    # 2+5 steps: 2 steps- current_node.right, = ; 5 steps -self.Node()
                self.createTree(temp[(med+1):, :], current_node.right, axis)  # T(n/2)+2 steps: 2 steps-current_node.right, temp[], T(n/2) steps - self.createTree()
    
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = T(n/2)+70 = logn+70
    # T(n) is O(n) = logn
    # -----------------------------------------------------------------------  
    
    
        def k_nearest_neighbor(self, k, target, current_root, k_nearest_heap):  # 1 step
            """
            function used to find the k nearest neighbor of a given target
            :param k: type int, indicates how many nearest neighbors to find
            :param target: type list, the target point
            :return: k_nearest_heap, type list
            """
            iter_list = []  # a stack to store iteration path          # 1 step
            # step1: find the 'nearest' leaf
            nearest_leaf = current_root                                # 1 step
            while nearest_leaf is not None:                            # 2 steps: while, is not
                iter_list.append(nearest_leaf)  # store the path       # 1 step
                tt = nearest_leaf.point                                # 2 steps: nearest_leaf.point, tt = 
                if target[nearest_leaf.axis] < nearest_leaf.point[nearest_leaf.axis]:  # 6 steps: if, <, nearest_leaf.axis, nearest_leaf.point, nearest_leaf.point[],target[]
                    if nearest_leaf.left is not None:  # then go to the left child     # 3 steps: if, is not, nearest_leaf.left
                        nearest_leaf = nearest_leaf.left                               # 2 steps: nearest_leaf.left, nearest_leaf = 
                    else:
                        break
                else:
                    if nearest_leaf.right is not None:   # else, go to the right child
                        nearest_leaf = nearest_leaf.right
                    else:
                        break
            while nearest_leaf.left is not None or nearest_leaf.right is not None:  # 6 steps: while, is not, or, is not, nearest_leaf.left, nearest_leaf.right
                if nearest_leaf.left is not None:                      # 3 steps: if, is not, nearest_leaf.left
                    nearest_leaf = nearest_leaf.left                   # 2 steps: nearest_leaf.left, = 
                    iter_list.append(nearest_leaf)                     # 1 step
                if nearest_leaf.right is not None:                     # 3 steps: if, is not, nearest_leaf.right
                    nearest_leaf = nearest_leaf.right                  # 2 steps: nearest_leaf.right, = 
                    iter_list.append(nearest_leaf)                     # 1 step
            tt = nearest_leaf.point                                    # 2 steps: nearest_leaf.point, tt = 
            """
            step2: find the k nearest by backtracking upside
            Two situations to add the point into the heap k_nearest_heap
            A. when len(k_nearest_heap) < k
            B. when dis(point, target) < current_max_dis
            """
            # k_nearest_heap = LargeHeap()  # the large heap to store the current 'nearest' neighbors
            # the max distance is actually the distance between target and the top of the heap
            '''
            current_max_dis = self.distance(target, nearest_leaf.point[:self.n_dim])
            k_nearest_heap.add(nearest_leaf, current_max_dis)
            tmp = iter_list.pop()
            '''
            former_node = nearest_leaf  # the former 'current_node', to indicate whether go through this child
                                                                # 1 step
            while iter_list != []:                              # 2 steps: while, !=
                if k_nearest_heap.len > 0:                      # 3 steps: if, k_nearest_heap.len, >
                    current_max_dis = k_nearest_heap.heaplist[0][1]  # 4 steps: k_nearest_heap.heaplist, k_nearest_heap.heaplist[0], k_nearest_heap.heaplist[0][1], current_max_dis =
                else:
                    current_max_dis = -1
                current_pointer = iter_list.pop()               # 1+38 steps: 1 step - current_pointer = ; 38 steps - iter_list.pop()
                tt = current_pointer.point                      # 2 steps: current_pointer.point, tt=
                dis = self.distance(current_pointer.point[:self.n_dim], target)  
                                                                # 1+11 steps: 1 step - dis=, 11 steps - self.distance()
                if k_nearest_heap.len < k:
                    k_nearest_heap.add(current_pointer, dis)
                elif dis < current_max_dis:                     # 2 steps: elif, <
                    k_nearest_heap.pop()                        # 38 steps: k_nearest_heap.pop()
                    k_nearest_heap.add(current_pointer, dis)    # 30 steps: k_nearest_heap.add()
                # current_max_dis = self.distance(k_nearest_heap.heaplist[0][0].point[:self.n_dim], target)
                current_max_dis = k_nearest_heap.heaplist[0][1] # 4 steps: k_nearest_heap.heaplist, k_nearest_heap.heaplist[],k_nearest_heap.heaplist[][], current_max_dis =
                axis = current_pointer.axis                     # 2 steps: current_pointer.axis, axis = 
                if abs(target[axis] - current_pointer.point[axis]) >= current_max_dis:
                                                                # 6 steps: if, >=, target[axis], - , current_pointer.point[], abs()
                    former_node = current_pointer               # 1 step
                    # if not intersect with
                    continue                                    # 1 step
                if current_pointer.left is not None and current_pointer.left != former_node:
                                                                # 5 steps: if, is not, and, current_pointer.left, !=
                    tt = current_pointer.left                   # 2 steps: current_pointer.left, tt =
                    # iter_list.append(current_pointer.left)
                    self.k_nearest_neighbor(k, target, current_pointer.left, k_nearest_heap)
                                                                # T(n/2) steps: self.k_nearest_neighbor()
                if current_pointer.right is not None and current_pointer.right != former_node:
                                                                # 5 steps: if, is not, and, current_pointer.left, !=
                    tt = current_pointer.right                  # 2 steps: current_pointer.left, tt =
                    # iter_list.append(current_pointer.righat)
                    self.k_nearest_neighbor(k, target, current_pointer.right, k_nearest_heap)
                                                                # T(n/2) steps: self.k_nearest_neighbor()
                former_node = current_pointer                   # 1 step
            rlist = []                                          # 1 step
            rdis = []                                           # 1 step
            for ele in k_nearest_heap.heaplist:                 # 2 steps: for, in 
                rlist.append(ele[0].point)                      # 3 steps: append(), ele[0], ele[0].point
                rdis.append(ele[1])                             # 2 steps: append(), ele[1]
            return rdis, rlist                                  # 1 step
        
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = T(n/2) + 211 = logn+211
    # T(n) is O(n) = logn
    # -----------------------------------------------------------------------     
        
        
        
        class Node:
            def __init__(self, point=None, axis=None):       # 1 step
                """
                :param point: type list, indicates a sample, not contain id&label
                :param axis: type int, the splitting axis on this splitting
                """
                # self.parent = parent
                self.left = None                             # 1 step
                self.right = None                            # 1 step
                self.point = point  # indicates the point sample for this node
                                                             # 1 step
                self.axis = axis   # indicates the splitting axis for this node
                                                             # 1 step
                # self.flag = 0   # flag used in traverse, to indicate whether visited, 0 means not visited
        
             
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 5
    # T(n) is O(n) = 5
    # -----------------------------------------------------------------------
        
        
        class LargeHeap:
            def __init__(self):  # 1 step
                self.len = 0     # 1 step
                self.heaplist = []   # here we use a list to store heap # 1 step
                
        
            def adjust(self):               # 1 step 
                # adjust to a large heap, assuming that only the last element in heaplist is not legal
                i = self.len - 1            # 2 steps: i=, -
                while i > 0:                # 2 steps: while, >
                    if self.heaplist[i][1] > self.heaplist[int((i - 1) / 2)][1]:
                                            # 7 steps: self.heaplist[i], self.heaplist[i][1], i-1, /2, int(), self.heaplist[], self.heaplist[][1]
                        self.heaplist[i], self.heaplist[int((i - 1) / 2)] = self.heaplist[int((i - 1) / 2)], self.heaplist[i]
                                            # 11 steps: self.heaplist[i]=, self.heaplist[int((i - 1) / 2)] =, i-1, /2, int(), self.heaplist[], i-1, /2, int(), self.heaplist[], self.heaplist[i]
                        i = int((i - 1) / 2)  # 4 steps: i-1, /2, int(), i = 
                    else:
                        break
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 27
    # T(n) is O(n) = 27
    # -----------------------------------------------------------------------  
                    
        
            def add(self, x, distance):      # 1 step
                """
                :param x: type Node, indicates a sample
                :param distance: type double, use to indicate "large"
                :return: None
                """
                # add a point and adjust it to a large heap
                self.len += 1                # 1 step
                self.heaplist.append([x, distance])   # append it to the end, and use adjust()
                                             # 1 step
                self.adjust()                # 27 steps: self.adjust()   
    
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 30
    # T(n) is O(n) = 30
    # -----------------------------------------------------------------------  
            def adjust2(self):                # 1 step
                # adjust to a large heap, assuming that only the first element(top) in heaplist is not legal
                i = 0                         # 1 step
                # attention to exchange with the large one of the children
                while (2*i + 1) < self.len:   # 4 steps: while, <, 2*i, +
                    if  (2*i+2 >= self.len):  
                        max_ind = (2*i + 1)
                    elif self.heaplist[(2*i + 1)][1] > self.heaplist[(2*i + 2)][1]:
                                              # 6 steps: elif, >, self.heaplist[],self.heaplist[][],self.heaplist[],self.heaplist[][]
                        max_ind = (2*i + 1)   # 3 steps: 2*i, +, max_ind = 
                    else:
                        max_ind=(2*i + 2)
                    if self.heaplist[i][1] < self.heaplist[max_ind][1]: 
                                              # 6 steps: if, <, self.heaplist[i],self.heaplist[i][1],self.heaplist[max_ind], self.heaplist[max_ind][1]
                        self.heaplist[i], self.heaplist[max_ind] = self.heaplist[max_ind], self.heaplist[i] 
                                              # 4 steps: self.heaplist[max_ind],  self.heaplist[i], self.heaplist[i] = , self.heaplist[max_ind] =
                        i = max_ind           # 1 step
                    else:
                        break
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 26
    # T(n) is O(n) = 26
    # -----------------------------------------------------------------------        
                    
        
            def pop(self):                 # 1 step
                # pop the top of the heap
                if self.len == 1:          # 2 steps: if, ==
                    self.heaplist = []     # 1 step
                    self.len = 0           # 1 step
                    return                 # 1 step
                # exchange for the last ele, and use adjust2()
                self.heaplist[0] = self.heaplist[-1]  # 3 steps: self.heaplist[-1], self.heaplist[0], = 
                self.len -= 1                         # 1 step
                self.heaplist = self.heaplist[:self.len]  # 2 steps: self.heaplist[:self.len], =
                self.adjust2()                        # 26 steps: self.adjust2()  
        
    # -----------------------------------------------------------------------
    # Total:
    # T(n) = 38
    # T(n) is O(n) = 38
    # -----------------------------------------------------------------------        
                    
        
    
    ##toString function to return the name of classifier
    #  @param self The object pointer
    def toString(self):                      # 1 step
        return "kdTreeKNNClassifier with k as "+str(self.k) # 2 steps: str(), return
    
    ##setK function to set k in the classifier type
    #  @param self The object pointer
    #  @param kvalue Value of k want to set
    def setK(self, kvalue):                  # 1 step
        self.k = kvalue                      # 1 step
        print("kvalue is saved")             # 1 step
    
    ##askK function to set k in the classifier type
    #  @param self The object pointer
    def askK(self):                         # 1 step
        kvalue = input("What is the kvalue you want to save?")  # 2 steps: input(), kvalue=
        self.k=kvalue                       # 1 step
        print("kvalue is saved")            # 1 step 
        
   ##override test function to return test set
    #  @param self The object pointer
    #  @param testData The test dataset
    #  @param k parameter for kNNkdTree
    #  @return the prediction result of kNNkdTree
    def test(self,testData,k=None):                 # 1 step
        
        super().test(testData)                      # 1 step
        
        if k == None:                               # 2 steps: if, ==
            k = self.k                              # 1 step, 1 space
        
        char_to_int = dict((c, i) for i, c in enumerate(self.trainL))          # 6 steps: enumerate(), for, in, (c,i), dic(), char_to_int()
                                
        self.trainL_Num= np.array([char_to_int[char] for char in self.trainL]) # 5 steps: for, in, [char_to_int[char]], np.array(), self.trainL_Num=
        
        tree = self.KdTree(n_dim=self.trainDS.shape[1])                        # 2logn + 286 + 2 steps: 2logn+ 286 steps - self.KdTree(); 2 steps - self.trainDS.shape[1], tree= 
        tree.createTree(np.c_[self.trainDS.astype(np.float), self.trainL_Num],self.KdTree.Node())
                                                                               # logn+70+3 steps: logn+70 steps - createTree(), 3 steps - astype(), self.KdTree.Node(), np.c_, 
        
        self.prediction = []                                                   # 1 step
                                                                               ## **-- p: len(self.testDS.shape[0]) --** ##
        for i in range(self.testDS.shape[0]):                                  # 3p+2 steps: 2 steps-self.testDS.shape[0], range(), 3p steps: i<range(), i++, i=
            classCounter = {}  # vote                                          # p steps
            dis, k_nearest = tree.k_nearest_neighbor(k, self.testDS.astype(np.float)[i], tree.root,self.KdTree.LargeHeap())
                                                                               # (logn+211+2+124)*p steps: logn+211 steps - k_nearest_neighbor(), 2 steps - self.testDS.astype(np.float)[i], tree.root, 124 steps - self.KdTree.LargeHeap()
            for pos in k_nearest:                                              # 2p steps for, in 
                classCounter[pos[-1]] = classCounter.get(pos[-1], 0) + 1       # 5p steps: pos[-1],get(),get()+1, classCounter[pos[-1]], = 
            self.prediction.append(sorted(classCounter)[0])                    # 3p steps: sorted(), sorted()[0], append
        
        int_to_char = {v: k for k, v in char_to_int.items()}                   # 6 steps: char_to_int.items(), for, in, v:k, {}, int_to_char=
        self.prediction= [int_to_char[char] for char in self.prediction]       # 5 steps: for, in, int_to_char[char], [], self.prediction=
        return self.prediction                                                 # 1 step
                
    # -----------------------------------------------------------------------
    # Total:
    # p: len(self.testDS.shape[0]) 
    # T(n) = (3+p)logn + 351p + 392
    # T(n) is O(n) = plogn
    # -----------------------------------------------------------------------        
 
 
                  
##################################################        
### ------------------------------------------ ###
## Total for kdTreeKNNClassifier:
## p: len(self.testDS.shape[0]) 
## T(n) =(3+p)logn + 351p + 404 
## T(n) is O(n) = plogn
### ------------------------------------------ ###  
##################################################  