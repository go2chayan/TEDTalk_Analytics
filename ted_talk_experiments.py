import ted_talk_sentiment as ts

# This python file is for enlisting all the experiments we are doing
# It can also be used as sample usage of the code repository such as
# the sentiment_comparator class.
###################################################################
# DO NOT delete an experiment even though they are highly redundant
###################################################################


# Check the average emotion plots
def bluemix_plot1():
    comparator = ts.Sentiment_Comparator(
        ts.hi_lo_files,     # Compare between hi/lo viewcount files
        ts.read_bluemix,    # Use bluemix sentiment
        )
    avg_ = comparator.calc_group_mean()

    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[0,1,2,3,4],   # only emotion scores
        styles=['r.--','r-','r--','r.-','ro-',
                'b.--','b-','b--','b.-','bo-'],  # appropriate line style
        legend_location='lower center'
        )

# Check the average social plots
def bluemix_plot2():
    comparator = ts.Sentiment_Comparator(
        ts.hi_lo_files,     # Compare between hi/lo viewcount files
        ts.read_bluemix,    # Use bluemix sentiment
        )
    avg_ = comparator.calc_group_mean()

    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[5,6,7],   # only social scores
        styles=['r.--','r-','r--',
                'b.--','b-','b--'],  # appropriate line style
        legend_location='lower center'
        )

# Check the average social plots
def bluemix_plot3():
    comparator = ts.Sentiment_Comparator(
        ts.hi_lo_files,     # Compare between hi/lo viewcount files
        ts.read_bluemix,    # Use bluemix sentiment
        )
    avg_ = comparator.calc_group_mean()

    # Plot Group Average
    ts.draw_group_mean_sentiments(avg_, # the average of groups
        comparator.column_names,        # name of the columns
        selected_columns=[8,9,10,11,12],   # only big5 scores
        styles=['r.--','r-','r--','r.-','ro-',
                'b.--','b-','b--','b.-','bo-'],  # appropriate line style
        legend_location='lower center'
        )

# bluemix plot one by one
def bluemix_plot4():
    comparator = ts.Sentiment_Comparator(
        ts.hi_lo_files,     # Compare between hi/lo viewcount files
        ts.read_bluemix,    # Use bluemix sentiment
        )
    avg_ = comparator.calc_group_mean()

    for i in range(13):
        # Plot Group Average
        ts.draw_group_mean_sentiments(avg_, # the average of groups
            comparator.column_names,        # name of the columns
            selected_columns=[i],   # only emotion scores
            styles=['r-',
                    'b-'],  # appropriate line style
            legend_location='lower center'
            )

# Checking time averages to see the best sentiment
def bluemix_plot5():
    comparator = ts.Sentiment_Comparator(
        ts.hi_lo_files,     # Compare between hi/lo viewcount files
        ts.read_bluemix,    # Use bluemix sentiment
        )
    avg_,p = comparator.calc_time_mean()

    ts.draw_time_mean_sentiments(avg_, # time averages
        comparator.column_names,       # name of the columns
        p                              # p values                      
        )

# See the sentences of a talk from a certain percent to another percent
def see_sentences():
    # Display sample sentences
    comparator.display_sentences(66, # Talk ID
        50, # Start percent
        60  # End percent
        )

if __name__=='__main__':
    # bluemix_plot1()
    # bluemix_plot2()
    # bluemix_plot3()
    #bluemix_plot4()
    bluemix_plot5()

