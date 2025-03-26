#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:45:31

@author: JM-RPC
"""
#import pdb; pdb.set_trace()
from sklearn.metrics import roc_curve, auc
import statsmodels.api  as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_partregress_grid, plot_leverage_resid2, influence_plot, plot_fit
from scipy import stats
import numpy as np
import pandas as pd
import io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import shinywidgets
from shinywidgets import render_widget, output_widget
import plotly.graph_objs as go
import plotly.express as pltx
import os
import signal
from datetime import datetime
from shinywidgets import output_widget, render_widget

max_factor_values = 25

basecolors0 = ['red',  'blue', 'green', 'goldenrod', 'violet','cyan', 'yellow','grey','gold','magenta','silver','orange','olive','khaki','thistle']
basecolorsalpha = ['red',  'blue', 'green', 'goldenrod', 'violet','cyan', 'yellow','grey','gold','magenta']
basecolors = [matplotlib.colors.to_rgba(item,alpha = None) for item in basecolorsalpha]
protected_names = ['Residuals','Predictions','Deviance_Resid','CI_lb', 'CI_ub','PI_lb', 'PI_ub']

def getcolor(col_data):
    dfc = pd.DataFrame(col_data).astype('str')
    choicesCo = list(dfc[dfc.columns[0]].astype('str').unique())
    choicesCo.sort()
    if (len(choicesCo) < len(basecolors)):
        colorD = {item : basecolors[choicesCo.index(item)]  for item in choicesCo}
        colorlist = [colorD[item] for item in col_data]
        lpatches = [mpatches.Patch(color = colorD[item],label = item) for item in colorD.keys()]
    else:
        cmap = plt.cm.plasma
        #colorNos = [choicesCo.index(item) for item in col_data]
        colorD = {item : cmap(choicesCo.index(item)/len(choicesCo)) for item in choicesCo}
        colorlist = [colorD[item] for item in col_data]
        lpatches = [mpatches.Patch(color = colorD[item],label = item) for item in choicesCo]
    return colorlist, lpatches, colorD

def collisionAvoidance(name,namelist):
    while name in namelist: 
        name = name + '_0'        
    return(name)

def doCorr(xv, yv, **kws):
    r,p = stats.pearsonr(xv,yv)
    ax = plt.gca()
    ax.annotate("r = {:.3f}, p = {:.3f}".format(r,p),xy=(.1, .9), xycoords=ax.transAxes)

app_ui = ui.page_navbar( 
    ui.nav_panel("Plotting",
                 ui.row(
                     ui.column(4,offset = 0,*[ui.input_file("file1", "Choose .csv or .dta File", accept=[".csv",".CSV",".dta",".DTA"], multiple=False, placeholder = '', width = "600px")]),
                     ui.column(3,offset = 0,*[ui.input_radio_buttons('killna', 'On input remove rows with missing data in one or more columns?',choices = ['No','Yes'],inline = True)]),
                     ui.column(2,offset = 0,),
                     ui.column(3,offset = 0,*[ui.a("Documentation", href="https://github.com/JM-RPC/plotit/blob/main/plotit_documentation.pdf",target="_blank")]),
                     ),
                 ui.row(
                     ui.output_text("plt_mess",inline = True)
                     ),
                 ui.row(
                     ui.input_radio_buttons("datachoose","Data:",choices = ['Input Data'], selected = 'Input Data', inline = True),
                     ),
                 ui.row(
                     ui.column(2,offset=0,*[ ui.input_selectize("xvar","X variable:",choices = ['-'], multiple=False)]),
                     ui.column(2,offset=0,*[ ui.input_selectize("yvar","Y variable:",choices = ['-'], multiple=False)]),
                     ui.column(2,offset=0,*[ ui.input_selectize("zvar","Z variable:",choices = ['-'], multiple=False)]),
                     ui.column(2,offset=0,*[ ui.input_selectize("cvar","Color with:",choices = ['-'], multiple=False)]),
                     ),
                 ui.row(
                     ui.column(1,offset=0,*[ui.input_action_button("updateB", "Update")]),
                     ui.column(1,offset=0),
                     ui.output_ui("pltopts"),
                     ui.column(1,offset=0),
                     ui.column(3,offset = 0,*[ui.download_button("downloadDP","Save Plotting Data",width = "200px")]),                
                     ),
                 ui.row(ui.output_ui("grphopts"),
                     ),
                 ),
    ui.nav_panel("Plotting Tools",
                ui.row(ui.HTML("<p><h4><b>Filters:</b> (filter on \"-\" to clear filter)</h4></p>")),
                ui.row(ui.HTML("<h4>Range Filter (numerical variables only, missing data will be filtered out):</h4>")),
                ui.row(
                    ui.input_selectize("rfvar","Filter On:" ,choices = ['-'], multiple=False),
                    ui.input_numeric("rflo", "Lower Bound:", value = 0.0, width = '300px'),
                    ui.input_numeric("rfhi", "Upper Bound:", value = 0.0, width = '300px'),
                    #ui.input_selectize("fitems","Included Rows:",choices = ['-'], multiple=True),
                    ),
                ui.row(
                    ui.HTML("<p>Current Active Ranges:</p>"),
                ),
                ui.row(
                    ui.output_text_verbatim("rlog")
                ),

                ui.row(
                    ui.HTML("<h4>Categorical Filter:</h4>")
                ),
                ui.row(
                    ui.input_selectize("fvar","Filter On:" ,choices = ['-'], multiple=False),
                    ui.input_selectize("fitems","Included Rows:",choices = ['-'], multiple=True),
                ),
                ui.row(
                    ui.HTML("<p>Rows Selected:</p>"),
                ),
                ui.row(
                    ui.output_text_verbatim("log")
                ),
                ui.row(
                    ui.HTML("<p><h4><b> Plot Customization: </b> </h4></p>")
                ),   
                ui.row(
                    ui.input_text("titleplt","Plot Title:", value = '-', width = '800px', )
                ),
                ui.row(ui.HTML("<p>These bounds override plot bounds.  To reset, reselect the variable.</p>")
                    ),
                ui.row(
                     ui.column(2,offset=0,*[ ui.input_numeric("xlb", "X lower bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("xub", "X upper bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("ylb", "Y lower bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("yub", "Y upper bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("zlb", "Z lower bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("zub", "Z upper bound:", value="",width=10)]),
                    ),                 
                ui.row(ui.HTML("<p><h4><b> Additional Data Series:</b></h4></p>"),
                      ),
                ui.row(
                       ui.column(2,offset=0,*[ ui.input_selectize("w1var","W1 variable:",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("w2var","W2 variable:",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("w3var","W3 variable:",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("w4var","W4 variable:",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("w5var","W5 variable:",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("w6var","W6 variable:",choices = ['-'], multiple=False)]),
                       ),
                ui.row(
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("w1mark","Type:",choices = ['dot','line'],inline = True)]),                    
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("w2mark","Type:",choices = ['dot','line'],inline = True)]),
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("w3mark","Type:",choices = ['dot','line'],inline = True)]),
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("w4mark","Type:",choices = ['dot','line'],inline = True)]),
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("w5mark","Type:",choices = ['dot','line'],inline = True)]),
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("w6mark","Type:",choices = ['dot','line'],inline = True)]),
                        ),
                ui.row(
                       ui.column(2,offset=0,*[ ui.input_selectize("w1col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("w2col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("w3col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("w4col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("w5col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("w6col","Color:",choices = basecolors0, multiple = False)]),                    
                       ),
                ui.row(
                       ui.column(2,offset=0,*[ ui.input_numeric("siglev", "Significance Level:", value="0.05",width=20)]),
                       ),
                ),
    
    ui.nav_panel("Data View",
                 #ui.input_file("file1", "Choose .csv or .dta File", accept=[".csv",".CSV",".dta",".DTA"], multiple=False, width = "500px", placeholder = ''),
                 #ui.output_text('io_mess'),ui.input_radio_buttons('killna', 'Remove rows with missing data in one or more columns?',choices = ['No','Yes']),
                 ui.row(ui.output_data_frame("info"), height = '300px'),
                 ui.row(ui.HTML("<p> </p>")),
                 ui.row(ui.output_data_frame("summary")),
                 ui.row(ui.HTML("<p> </p>")),
                 ui.row(ui.output_data_frame("data")),
                 ),

    ui.nav_panel("Correlations",
                   ui.row(ui.input_selectize("corrV","Select variables:",choices = [''],multiple = True,width = "200px")),
                   ui.output_plot("dataPD",width = '1200px', height = '1200px'),
                 ),
    ui.nav_panel("Linear Models",
                 ui.row(
                     ui.column(6,offset=0,*[ui.input_radio_buttons("mtype","Model Type",choices = ['OLS','LOGIT','POISSON','NEGATIVE_BINOMIAL'],inline = True)]),
                     ui.column(3,offset = 0,*[ui.output_ui("nbparm")]),
                     ui.column(3,offset = 0),
                     ),
                 ui.row(
                     ui.column(3, offset = 0, *[ui.input_selectize("depvar","Dependent Variable:",choices = ['-'],multiple = False)]),
                     ui.column(5, offset = 0, *[ui.input_selectize("indvar","Independent Variables:", choices = ['-'],multiple = True, width = "600px")]),
                     ui.column(4, offset = 0, *[ui.input_selectize("tofactor","Convert Numeric Variables to factors:", choices = ['-'],multiple = True)]),
                     ),
                 ui.row(ui.HTML("<p> Use Wilkinson/Patsy notation to specify variable transformations.</p>")
                     ),
                 ui.row(
                     ui.column(2, offset = 0, *[ui.input_action_button('modelGo',"Run Model")]),
                     ui.column(10, offset = 0, *[ui.input_text('stringM','Model String:',width = '1000px')]),
                     #ui.column(2, offset = 0, *[ui.input_action_button('modelClear',"Clear Model")]),
                     ),
                 ui.row(
                     ui.output_ui("dloads"),
                     ),
                 ui.row(
                     ui.output_text_verbatim("modelSummary")
                     ),
                 ),
    ui.nav_panel("Linear Models: Standard Plots",
                  ui.row(
                      ui.column(6,offset=0,*[ui.input_radio_buttons("regplotR","Plot: ",choices = ['ROC', 'Leverage','Partial Regression','Influence','Fit'],inline = True)]),
                      ui.column(6,offset=0,*[ui.input_select("lmsp","Fit: Ind. Var:",choices = ['-'])]),
                      ),
                  ui.row(
                      ui.output_plot("regplot1", width = "900px", height = "600px")
                      ),
                  # ui.row(
                  #     ui.output_plot("regplot3", width = "900px", height = "600px")
                  #     ),
                  ),
    ui.nav_panel("Log",
                  ui.input_action_button("logGo","Update Log"),
                  ui.download_button("logdown","Download Log"),
                  ui.input_action_button("resetlog", "Reset Log"),
                  ui.output_text_verbatim("logout")
                  ),
    ui.nav_spacer(),
    ui.nav_menu("Exit",
                 ui.nav_control(
                     ui.column(1,offset=0,*[ui.input_action_button("exit","Exit")]),
                     ),
                 ),
    
               
underline = True, title = "plotit v.0.0.3 ")

def server(input: Inputs, output: Outputs, session: Session):
    mdl_type = reactive.value("OLS") #currently supported types: OLS, LOGIT
    mdl = reactive.value(None)
    mdl_depvar = reactive.value('-')
    mdl_indvar = reactive.value(())
    mdl_stringM = reactive.value(" - ~ ")
    mdl_data = reactive.value(pd.DataFrame())
    subdict = reactive.value({}) #dictionary containing active row entries for each factor column column (maximum of max_factor_values unique entries.)
    rngdict = reactive.value({}) #dictionary containing active ranges for each numerical column
    logstr = reactive.value("")
    dbgstr = reactive.value(f"At server start: Figures: {plt.get_fignums()} \n")
    plt_msgstr = reactive.value("")
    io_msgstr = reactive.value("")
    lm_msgstr = reactive.value("")
    plt_data = reactive.value(pd.DataFrame())
    logstr = reactive.value(f"Log Start: {datetime.now()}")
    trendOn = reactive.value(False)

    @reactive.effect
    @reactive.event(input.exit)
    async def do_exit():
        #plt.close(fig)
        await session.app.stop()
        os.kill(os.getpid(), signal.SIGTERM)
        max_factor_values = 50

        
##########################################################################
####  Log panel
##########################################################################
    @render.text  
    @reactive.event(input.logGo)
    def logout():
        return logstr()
    
    @reactive.effect
    @reactive.event(input.resetlog)
    def resetL():
        logstr.set(f"Log Start: {datetime.now()}")
        return
    
    def pushlog2(newlogstr):
        #print(newlogstr)
        # with reactive.isolate():
        #     logstr.set(logstr() + '\n' + '***' + newlogstr)
        return    
    
    def pushlog(newlogstr):
        with reactive.isolate():
            logstr.set(logstr() + '\n'+ newlogstr)
        return

    @render.download(filename = "Logfile.txt")
    def logdown():
        loglst = logstr().splitlines()
        dflog = pd.DataFrame(loglst)
        yield dflog.to_csv(index = False)

        
##########################################################################
####  Input panel
##########################################################################

    @reactive.calc
    def parsed_file():
        if input.file1() is None:
            return pd.DataFrame()
        else: #Note the ui passes only paths ending in  .csv, .CSV, .dta, and .DTA
            fpath = str(input.file1()[0]['datapath'])
            if (fpath[-4:] == '.csv') or (fpath[-4:] == '.CSV'):
                #use this to create an option for extra large files to guess types from the first 1000 rows
                ##df1 = pd.read_csv(input.file1()[0]["datapath"],nrows = 1000)
                ##dataDict = dict(zip(list(df1.dtypes.index),list(df1.dtypes)))
                ##df = pd.read_csv(input.file1()[0]["datapath"],dtype = dataDict)
                df = pd.read_csv(input.file1()[0]["datapath"])
            else:
                df = pd.read_stata(input.file1()[0]["datapath"])
            pushlog("************************************************")
            pushlog("File read: "  + input.file1()[0]['name'])
            pushlog(f"....Number of rows: {len(df)}")
            pushlog("************************************************")
            stemp = df.isna().sum().sum()
            df.replace('',np.nan,inplace = True)
            stemp = df.isna().sum().sum() - stemp
            nona = sum(df.isna().sum(axis=1) >0)
            if (stemp > 0) | (nona > 0):
                #io_msgstr.set(f" {stemp} blank entries converted to NaNs. {nona} rows out of {len(df)} have missing data.")
                pushlog(f" {stemp} blank entries converted to NaNs. {nona} rows out of {len(df)} have missing data.")
                
            #get rid of spaces in column names
            df.columns = df.columns.str.lstrip()
            df.columns = df.columns.str.rstrip()
            df.columns = df.columns.str.replace(' ','_')
            #change names to avoid collisions with protected names
            df.columns = [collisionAvoidance(item,protected_names) for item in df.columns]
            if (input.killna() == 'Yes') : 
                pushlog("Rows with missing values dropped on input by user request.")
                df.dropna(inplace = True)
            #reset plotting data
            plt_data.set(df)
            #reset current model
            with reactive.isolate():#reset any current linear model
                pushlog2(f"In parsed_file, mdl_stringM() = {mdl_stringM()} resetting model.")
                mdl.set(None) #reset statsmodels result
                ui.update_selectize('depvar',selected = []) #reset dependent variable choices
                ui.update_selectize('indvar',selected = '-') #rset independent variable choices
                ui.update_radio_buttons('datachoose', selected = 'Input Data')
                ui.update_radio_buttons('mtype',selected = 'OLS')
                mdl_stringM.set('- ~') #reset model string
                mdl_data.set(pd.DataFrame()) #reset the model dataset
            return df

    @render.data_frame
    def info():
        #print("Render Info")
        df = parsed_file()
        #df = plt_data()
        if df.empty:
           return 
        #display df.info
        buffer = io.StringIO()
        df.info(buf=buffer)
        slst = buffer.getvalue().splitlines()
        sdf = pd.DataFrame([item.split() for item in slst[5:-2]],columns = slst[3].split())
        return sdf
    
    @render.data_frame
    def summary():
        #print("Render Data Frame summary")
        df = parsed_file()
        #df = plt_data()
        if df.empty:
            return pd.DataFrame()
        elif len(df) > 500000:
            return pd.DataFrame({'': ['Too many rows','Use \"summary() \" from command line.']},index = ['Problem: ','Solution: '])

        description = df.describe(include= "all")
        dindex = description.index.values
        description.insert(0," ",dindex)
        return description
    
    @render.data_frame
    def data():
        #print("Render Data Frame")
        df = parsed_file()
        if len(df) > 100000 : return
        return df
    
##########################################################################
####  Correlations panel
##########################################################################
    
    @render.plot
    @reactive.event(input.corrV)
    def dataPD():
        #print("Setup correlations")
        if input.corrV() == (): return
        df = plt_data()
        fig = plt.figure(figsize = (9,9))
        showC = input.corrV()
        showC = [item for item in showC]
        dfc = df[showC].copy()
        dfc.dropna(inplace = True)
        nobs = len(dfc)
        ax = sb.PairGrid(dfc, vars = showC, corner = True).set(title = f"# Obs.= {nobs}")
        ax.map_diag(sb.histplot, kde=True)
        ax.map_lower(plt.scatter, s= 2)
        ax.map_lower(doCorr)
        return 

##########################################################################
####  Plotting panel
##########################################################################
    @reactive.effect
    @reactive.event(input.datachoose)
    def data_update():
        pushlog("data_update")
        with reactive.isolate():
            if (input.datachoose() == 'Model Data') :
                pushlog("...Switching to model data.")
                plt_data.set(mdl_data())
            if (input.datachoose() == 'Input Data'):
                pushlog("...Switching to original input data.")
                ui.update_radio_buttons("datachoose",label = "Data:",choices = ["Input Data"])
                plt_data.set(parsed_file())


    @reactive.effect
    @reactive.event(parsed_file, input.datachoose)
    def setupPlot():
        pushlog("Initializing Plotting Data (setupPlot)")
        df = pd.DataFrame()
        df = plt_data()
        nrow = len(df)
        if (nrow == 0): 
            pushlog(f"...{nrow} rows in current data, data source = {input.datachoose()}")
            return
        cols = list(df.columns)
        num_var = list(df.select_dtypes(include=np.number).columns)
        str_var = [item for item in cols if item not in num_var]   
        pushlog2(f"Numerical vars: {'; '.join(num_var)}") 
        pushlog2("...")
        pushlog2(f"String  vars: {'; '.join(str_var)}") 
        pushlog2("...")
        #fct used for subsetting (fct short for factor) and coloring
        fct_var = [item for item in cols if ((item not in num_var) or (len(list(df[item].unique()))<=max_factor_values))]
        pushlog2(f"Factor vars: {'; '.join(fct_var)}") 
        #subset dictionary
        pushlog2("...Initializing filters.")
        newdict = {}                
        newdict = {item: list(map(str,list(df[item].unique()))) for item in fct_var}
        subdict.set(newdict)
        num_fct = [item for item in list(df.columns) if (item in num_var) and len(list(df[item].unique())) <= max_factor_values]
        newRdict = {}
        newRdict = {item: [df[item].min(),df[item].max()] for item in num_var}
        rngdict.set(newRdict)
        ui.update_numeric("rfvar", value = '-')
        pushlog("...Initializing plotting data.")
        ui.update_selectize("xvar",choices = ['-']+num_var)
        ui.update_selectize("yvar",choices = ['-']+num_var)
        ui.update_selectize("zvar",choices = ['-']+num_var)
        ui.update_selectize("cvar",choices = ['-']+fct_var)
        ui.update_selectize("fvar",choices = ['-']+fct_var)
        ui.update_selectize("rfvar",choices = ['-'] + num_var)
        ui.update_selectize("corrV", choices = num_var, selected = None)
        #if (input.datachoose() == 'Input Data'):
        if 1 >0 :
            pushlog("... initializing model data")
            if (input.datachoose() == "Input Data"):
                ui.update_selectize("indvar",choices = num_var + str_var)
                ui.update_selectize("depvar",choices =  ['-'] + num_var)
                ui.update_selectize("tofactor",choices =  num_fct) 
                mdl.set(None)
            else: 
                ui.update_selectize("indvar", choices = num_var + str_var,selected = list(mdl_indvar()))
                ui.update_selectize("depvar", choices =  ['-'] + num_var,selected = mdl_depvar())
                ui.update_selectize("tofactor",choices =  num_fct,selected = list(input.tofactor())) 

        pushlog("...Initializing extra variables.")
        ui.update_selectize("w1var",choices = ['-'] + num_var)        
        ui.update_selectize("w2var",choices = ['-'] + num_var)        
        ui.update_selectize("w3var",choices = ['-'] + num_var)        
        ui.update_selectize("w4var",choices = ['-'] + num_var)        
        ui.update_selectize("w5var",choices = ['-'] + num_var)        
        ui.update_selectize("w6var",choices = ['-'] + num_var)      
        pushlog2("...setupPlot returning")  
        return
    


    @reactive.effect    
    @reactive.event(input.xvar)
    def do_xvar():
        if input.xvar() == '-': 
            plt.clf()
            return
        df = plt_data()
        ui.update_numeric("xlb",value = min(df[input.xvar()]))
        ui.update_numeric("xub",value = max(df[input.xvar()]))

    @reactive.effect    
    @reactive.event(input.yvar)
    def do_yvar():
        if input.yvar() == '-': 
            plt.clf()
            return
        df = plt_data()
        #ui.update_slider('sl1',label="Dot Size:",min=0.25,max=10.0,value=2.0,step=0.25)
        ui.update_numeric("ylb",value = min(df[input.yvar()]))
        ui.update_numeric("yub",value = max(df[input.yvar()]))
        
    @reactive.effect    
    @reactive.event(input.zvar)
    def do_zvar():
        if input.zvar() == '-': 
            plt.clf()
            return
        df = plt_data()
        ui.update_numeric("zlb",value = min(df[input.zvar()]))
        ui.update_numeric("zub",value = max(df[input.zvar()]))  

        
    # @render.text
    # def debug():
    #     return(dbgstr())
        
    #adjust ui to reflect number of variables (x only histogram/boxplot)   x and y or x,y and z scatterplot 
    @render.ui
    @reactive.event(input.xvar, input.yvar,input.zvar)
    def pltopts():
        df = plt_data()
        mxbin = len(df)
        if ((input.yvar() != '-') & (input.xvar() != '-') & (input.zvar() != '-')):
            if (input.datachoose() == 'Model Data')  & (set(input.indvar()) == set([input.xvar(),input.yvar()])) & (input.depvar() == input.zvar()):
                trendOn.set(True)
                return ui.TagList(
                              #ui.column(1,offset = 0,*[ui.input_action_button("updateB3", "Update")]),
                              ui.column(3,offset=0,*[ui.input_slider("sl1","# dotsize",min = 0, max = 75, value = 10)]),
                              ui.column(1,offset = 0,),
                              ui.column(4,offset=0,*[ui.input_checkbox_group("scttropts","3D Scatter Plot Options:",
                                                choices=('Show Trend','CI','PI'),selected=(),inline = True)])
                              )
            else:
                trendOn.set(False)
                return ui.TagList(
                              #ui.column(1,offset = 0,*[ui.input_action_button("updateB12", "Update")]),
                              ui.column(3,offset=0,*[ui.input_slider("sl1","# dotsize",min = 0, max = 75, value = 10)]),
                              #ui.column(1,offset = 0,),
                              #ui.column(2,offset=0,*[ui.input_checkbox_group("scttropts","Scatter Plot Options:",
                              #                                      choices=('Show Trend','CI','PI'),selected=(),inline = True)])
                              )
        elif (input.yvar() != '-') & (input.xvar() != '-'):  
            if (input.datachoose() == 'Model Data') & (set([input.xvar()])== set(input.indvar())) & (input.yvar() == input.depvar()):
                trendOn.set(True)
                return ui.TagList(
                              #ui.column(1,offset = 0,*[ui.input_action_button("updateB12", "Update")]),
                              ui.column(3,offset=0,*[ui.input_slider("sl1","# dotsize",min = 0, max = 75, value = 10)]),
                              ui.column(1,offset = 0,),
                              ui.column(2,offset=0,*[ui.input_checkbox_group("scttropts","Scatter Plot Options:",
                                                                    choices=('Show Trend','CI','PI'),selected=(),inline = True)])
                              )
            else:
                trendOn.set(False)
                return ui.TagList(
                              #ui.column(1,offset = 0,*[ui.input_action_button("updateB12", "Update")]),
                              ui.column(3,offset=0,*[ui.input_slider("sl1","# dotsize",min = 0, max = 75, value = 10)]),
                              #ui.column(1,offset = 0,),
                              #ui.column(2,offset=0,*[ui.input_checkbox_group("scttropts","Scatter Plot Options:",
                              #                                      choices=('Show Trend','CI','PI'),selected=(),inline = True)])
                              )               
        elif(input.xvar() != '-'):
            return ui.TagList(
                              #ui.column(1,offset = 0,*[ui.input_action_button("updateB12", "Update")]),
                              ui.column(3,offset=0,*[ui.input_radio_buttons("rb1","Plot type:",choices = ['Histogram','Boxplot','Kernel Density'],selected = 'Histogram',inline=True)]),
                              ui.column(2,offset=0,*[ui.input_numeric("sl1","# Bins",value=min(max(round(mxbin**0.33,0),10),50), width=2)])
                              )
        else:
            return None
    
    @render.ui
    @reactive.event(input.updateB, input.zvar)
    def grphopts():
        if(input.zvar()!= '-') :
            return ui.TagList(ui.column(12,offset=0,*[output_widget("plot3")])
                              )
        else:
            return ui.TagList(ui.column(12,offset=0,*[ui.output_plot("Plots", width = '900px', height = '600px')])
                              )  
        
    def doTrend():
        dfg = plt_data()
        with reactive.isolate():
            cur_mdl = mdl()
            xv = input.xvar()
            yv = input.yvar()
            zv = input.zvar()
            res = None
            MTYPE = None
            #set up the input data for the independent variables
            sq = 0.05
            gridcount = 25
            if (xv != '-'):
                deltax = dfg[xv].max() - dfg[xv].min()
                xlo = dfg[xv].min() - sq*deltax
                xup = dfg[xv].max() + sq*deltax
            if (yv != '-'): 
                deltay = dfg[yv].max() - dfg[yv].min()
                ylo = dfg[yv].min() - sq*deltay
                yup = dfg[yv].max() + sq*deltay
            if (zv != '-'): #We are doing 3D
                xvar, yvar = np.meshgrid(np.arange(xlo,xup,deltax/gridcount),np.arange(ylo, yup,deltay/gridcount))                
                exog0 = pd.DataFrame({xv: xvar.ravel(), yv: yvar.ravel()}) 
            else: #we are doing 2D
                xvar = np.arange(xlo, xup, deltax/gridcount) 
                yvar = []                      
                exog0 = pd.DataFrame({xv : xvar})
            #the data is set now either find an extant model or fit a model
            #if the x, y and z variables perfectly match the variables in the most recently estimated model, then use that model's results 
            #otherwise fit a new model
            depV = "".join(mdl_stringM().split()).split('~')[0]
            indV = "".join(mdl_stringM().split()).split('~')[1].split('+')
            res = mdl()
            MTYPE = mdl_type()
            try:
                res_predictions = res.get_prediction(exog=exog0,transform = True)
                res_frame = res_predictions.summary_frame(alpha = input.siglev())
            except:
                pushlog("...Predictions failed!")
                pushlog2("...Prediction data: ")
                pushlog2(f"...  {exog0.head()}")
                pushlog("...In order to plot a trend. \n ...Plotting variables and model variables must by identical \n (1 variable and 2 variable models only)")
                return [],[],[],[],[],[],[]
            znew = res_frame['mean'].values.reshape(xvar.shape)    
            Ci_lb1 =  res_frame['mean_ci_lower'].values.reshape(xvar.shape)
            Ci_ub1 =  res_frame['mean_ci_upper'].values.reshape(xvar.shape)
            if (MTYPE == 'OLS'):
                Pi_lb1 =  res_frame['obs_ci_lower'].values.reshape(xvar.shape)
                Pi_ub1 =  res_frame['obs_ci_upper'].values.reshape(xvar.shape)
            else:
                Pi_lb1 = []
                Pi_ub1 = []
            return xvar, yvar, znew, Ci_lb1, Ci_ub1, Pi_lb1, Pi_ub1
   
    @output
    @render_widget
    @reactive.event(input.updateB)
    def plot3():
        pushlog2("plot3()")
        plt.clf()
        if (input.xvar() == '-') | (input.yvar() == '-') | (input.zvar() == '-'): return
        df = plt_data()
        if len(df) == 0: 
#            plt_msgstr.set("You need a data set before you can plot.")
            return
        xv = input.xvar()
        yv = input.yvar()
        zv = input.zvar()
        w1v = input.w1var()
        w2v = input.w2var()
        w3v = input.w3var()
        w4v = input.w4var()
        w5v = input.w5var()
        w6v = input.w6var()
        totrow = len(df)
        cv = input.cvar()
        #print(f"Plotting: x = {xv}, y={yv}, z={zv}, color = {cv} ")
        if cv == '-': cv = None 
         #take out the rows that the user has decided to ignore
        for item in list(subdict().keys()):
            df = df[df[item].astype('str').isin(list(subdict()[item]))]
        for item in list(rngdict().keys()):
            df = df[(df[item]>=rngdict()[item][0]) & (df[item]<=rngdict()[item][1])]
        nrow0 = len(df)
        dfg = df.dropna(subset = [xv, yv, zv]) #get rid fo rows with na's in the columns to be plotted
        nrow = len(dfg)
        #plotly can only plot up to about 400,000 to 450,000 rows of data, randomly downsample if needed.  Warn user in the title & log
        if nrow > 400000:
            dfg = dfg.sample(400000)
            nrow = len(dfg)
            ttlstr = f"File: {input.file1()[0]['name']} <br> down-sampled!  {nrow} rows plotted out of {totrow} "
            pushlog("...Plotly 3D plotting limit warning! Attempted to plot {nrow} rows, downsampled to 400,000 rows.")
        else:
            ttlstr = f"File: {input.file1()[0]['name']}:  {nrow} rows plotted out of {totrow} "
            pushlog(f"...plot3: {totrow-nrow0} rows filtered out, {nrow0-nrow} rows removed due to missing data.")
        if input.titleplt() != "-" : 
            ttlstr = input.titleplt()
        if (cv != None):
            nucolor ,nupatches, colorD = getcolor(list(dfg[cv].astype('str'))) #fix tup color map
            colormap = {str(item) : f"rgb({int(250*float(colorD[item][0]))},{int(250*float(colorD[item][1]))},{int(250*float(colorD[item][2]))})" for item in colorD.keys() }
            fig = pltx.scatter_3d(dfg,x=xv, y= yv, z = zv, color = list(dfg[cv].astype('str')), color_discrete_map = colormap, width = 900, height = 600, title = ttlstr)
            #fig = go.Figure(data = go.Scatter3d(x=dfg[xv],y=dfg[yv],z=dfg[zv],mode='markers',marker= dict(color=nucolor),showlegend = True))
        else:
            fig = go.Figure(data = go.Scatter3d(x=dfg[xv],y=dfg[yv],z=dfg[zv],mode='markers'))
        fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}
        fig.update_layout(autosize = False,width = 900, height=900,title = ttlstr)
        fig.update_layout(scene = dict( xaxis_title = xv, yaxis_title = yv, zaxis_title = zv))
        fig.update_traces(marker = dict(size = int(input.sl1()/5 +1))) # fix up dot size
        if trendOn():
            if  "Show Trend" in input.scttropts(): 
                xvars, yvars, znew, Ci_lb1, Ci_ub1, Pi_lb1, Pi_ub1 = doTrend() 
                if len(znew) >0:                                
                    #we are ready to add the traces
                    fig.add_trace(go.Surface(x=xvars,y=yvars,z=znew, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                    if ('CI' in input.scttropts()):
                        fig.add_trace(go.Surface(x=xvars,y=yvars,z=Ci_lb1, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                        fig.add_trace(go.Surface(x=xvars,y=yvars,z=Ci_ub1, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                    if ('PI' in input.scttropts()) & (len(Pi_lb1) >0):
                        fig.add_trace(go.Surface(x=xvars,y=yvars,z=Pi_lb1, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                        fig.add_trace(go.Surface(x=xvars,y=yvars,z=Pi_ub1, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                else:
                    pushlog("...Trend calculation failed. Model variables and plotting variables must match.  Check and re-run model.")           
        if (w1v != '-'):
            dfg = dfg.dropna(subset = [w1v]) #get rid fo rows with na's in the columns to be plotted
            if (input.w1mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[w1v],mode = 'markers', marker = dict(size = int(input.sl1()/5 +1),color = input.w1col())))
                pushlog2(f"...3D extra plot #1 variable= {input.w1var()} color = {input.w1col()} NOTE: missing values are dropped.")
            elif (input.w1mark() == 'line'):
                pushlog2(f"...3D extra plot #1  variable= {input.w1var()} color = {input.w1col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[w1v],color = input.w1col(), opacity = 0.6))
        if (w2v != '-'):
            dfg = dfg.dropna(subset = [w2v]) #get rid fo rows with na's in the columns to be plotted
            if (input.w2mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[w2v],marker = dict(size = int(input.sl1()/5 +1),color = input.w1col())))
                pushlog2(f"...3D extra plot #2 variable= {input.w2var()} color = {input.w2col()} NOTE: missing values are dropped.")
            elif (input.w2mark() == 'line'):
                pushlog2(f"...3D extra plot #2  variable= {input.w2var()} color = {input.w2col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[w2v],color = input.w1col()))
        if (w3v != '-'):
            dfg = dfg.dropna(subset = [w3v]) #get rid fo rows with na's in the columns to be plotted
            if (input.w3mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[w3v],marker = dict(size = int(input.sl1()/5 +1),color = input.w1col())))
                pushlog2(f"...3D extra plot #3 variable= {input.w3var()} color = {input.w3col()} NOTE: missing values are dropped.")
            elif (input.w3mark() == 'line'):
                pushlog2(f"...3D extra plot #3  variable= {input.w3var()} color = {input.w3col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[w3v],color = input.w1col()))
        if (w4v != '-'):
            dfg = dfg.dropna(subset = [w4v]) #get rid fo rows with na's in the columns to be plotted
            if (input.w4mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[w4v],marker = dict(size = int(input.sl1()/5 +1),color = input.w1col())))
                pushlog2(f"...3D extra plot #4 variable= {input.w4var()} color = {input.w4col()} NOTE: missing values are dropped.")
            elif (input.w4mark() == 'line'):
                pushlog2(f"...3D extra plot #4  variable= {input.w4var()} color = {input.w4col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[w4v],color = input.w1col()))
        if (w5v != '-'):
            dfg = dfg.dropna(subset = [w5v]) #get rid fo rows with na's in the columns to be plotted
            if (input.w5mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[w5v],marker = dict(size = int(input.sl1()/5 +1),color = input.w1col())))
                pushlog2(f"...3D extra plot #5 variable= {input.w5var()} color = {input.w5col()} NOTE: missing values are dropped.")
            elif (input.w5mark() == 'line'):
                pushlog2(f"...3D extra plot #5  variable= {input.w5var()} color = {input.w5col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[w5v],color = input.w1col()))
        if (w6v != '-'):
            dfg = dfg.dropna(subset = [w6v]) #get rid fo rows with na's in the columns to be plotted
            if (input.w6mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[w6v],marker = dict(size = int(input.sl1()/5 +1),color = input.w1col())))
                pushlog2(f"...3D extra plot #6 variable= {input.w6var()} color = {input.w6col()} NOTE: missing values are dropped.")
            elif (input.w6mark() == 'line'):
                pushlog2(f"...3D extra plot #6  variable= {input.w6var()} color = {input.w6col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[w6v],color = input.w1col()))
        return fig
                
    @render.plot
    @reactive.event(input.updateB)
    def Plots():
        pushlog("Plots")
        if (input.zvar() != '-') : return # plotting happening in plot3()
        if (input.xvar() == '-'):
            plt.clf()
            return
        plt.clf()
        df = plt_data()
        if len(df) == 0: 
#            plt_msgstr.set("You need a data set before you can plot.")
            return
        xv = input.xvar()
        yv = input.yvar()
        zv = input.zvar()
        totrow = len(df)
        cv = input.cvar()
        pushlog2(f"...plotting x = {xv}, y = {yv}, color = {cv}")
        #print(f"Beginning of plots ...Model: {mdl_stringM()} indvar = {mdl_indvar()}  depvar = {mdl_depvar()}")
        #expand the plot axes a squidge for esthetics
        squidgeVal = 0.05
        if (xv != '-'):
            squidgeX = (input.xub()-input.xlb())*squidgeVal
        if (yv != '-'):
            squidgeY = (input.yub()-input.ylb())*squidgeVal
        #create the row subset for plotting     
        for item in list(subdict().keys()):
            df = df[df[item].astype('str').isin(list(subdict()[item]))]
        nrow0 = len(df)
        for item in list(rngdict().keys()):
            df = df[(df[item]>=rngdict()[item][0]) & (df[item]<=rngdict()[item][1])]
        nrow1 = len(df)
        pushlog(f"...{totrow-nrow0} rows excluded via category filter,  {nrow0-nrow1} via value filter.")
        #drop rows that have NaN values in the columns to be plotted
        sbst = [xv]
        if yv != '-': sbst.append(yv)
        if zv != '-': sbst.append(zv)
        dfg = df.dropna(subset = sbst)
        nrow = len(dfg)
        pushlog(f"...{nrow1 - nrow} rows dropped from plotting set due to missing data.")
        if (cv != '-'):
            colorlist,lpatches,colorD = getcolor(list(dfg[cv].astype('str')))
            color_data = dfg[cv].astype('str')
        else:
            cv = None
            colorlist = None
            color_data = None    
        fig = plt.figure(2,figsize = (12,9), tight_layout = False)
        fig.clf()
        #make this a user choice later
        sb.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        #initialize some parameters
        edgecol = 'black'
        kdeflag = False
        titlestr = f'File: {input.file1()[0]["name"]} rows shown {nrow} out of {totrow}'        
        if  ((xv != '-') & (yv != '-')):
            #numx = dfg[xv].unique
            ax=sb.scatterplot(data = dfg,x = xv,y = yv, c = colorlist,  s = input.sl1(),label = yv)
            if yv == 'Residuals':
                plt.axhline(y = 0, color = 'black')
            #print(f".....Just before Show Trend in plots mdl_indvar = {list(mdl_indvar())}, model: {mdl_stringM()}, input.indvar = {list(input.indvar())}, input.depvar = {input.depvar()}")
            if trendOn():
                if "Show Trend" in input.scttropts():
                    xvar, yvar, ynew, Ci_lb1, Ci_ub1, Pi_lb1, Pi_ub1 = doTrend()
                    if len(ynew) >0: 
                            ax = sb.lineplot(x=xvar, y= ynew, color = 'red') 
                            if ('CI' in input.scttropts()):
                                ax = sb.lineplot(x = xvar, y= Ci_lb1, color = 'green')
                                ax = sb.lineplot(x = xvar, y= Ci_ub1, color = 'green')
                            if ('PI' in input.scttropts()) & (len(Pi_lb1) > 0):
                                ax = sb.lineplot(x = xvar, y= Pi_lb1, color = 'goldenrod')
                                ax = sb.lineplot(x = xvar, y= Pi_ub1, color = 'goldenrod')   
                    else: 
                        pushlog("...Trend calculation failed. Plotting variables and model variables must match.  Check data and re-run model. ")  

            # plot extra series if needed                    
            if (input.w1var() != '-'):
                if (input.w1mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.w1var(), c = input.w1col(),  s = input.sl1(),label = input.w1var())
                    pushlog2(f"... extra plot #1 variable= {input.w1var()} color = {input.w1col()} NOTE: missing values are dropped.")
                    #ax.add(sb.Dot())    
                elif (input.w1mark() == 'line'):
                    pushlog2(f"... extra plot #1  variable= {input.w1var()} color = {input.w1col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.w1var(), color = input.w1col(),label = input.w1var())
                    #plt.legend(title = input.w1var(), loc = 1, fontsize = 12)
            if (input.w2var() != '-'):
                if (input.w2mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.w2var(), c = input.w2col(),  s = input.sl1(),label = input.w2var())
                    pushlog2(f"... extra plot #2 variable= {input.w2var()} color = {input.w2col()} NOTE: missing values are dropped.")  
                elif (input.w2mark() == 'line'):
                    pushlog2(f"... extra plot #2 variable= {input.w2var()} color = {input.w2col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.w2var(), color = input.w2col(),label = input.w2var())
                    #plt.legend(title = input.w2var(), loc = 1, fontsize = 12)
            if (input.w3var() != '-'):
                if (input.w3mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.w3var(), c = input.w3col(),  s = input.sl1(),label = input.w3var())
                    pushlog2(f"... extra plot #3 variable= {input.w3var()} color = {input.w3col()} NOTE: missing values are dropped.")  
                elif (input.w3mark() == 'line'):
                    pushlog2(f"... extra plot #3 variable= {input.w3var()} color = {input.w3col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.w3var(), color = input.w3col(),label = input.w3var())
                    #plt.legend(title = input.w3var(), loc = 1, fontsize = 12)                    
            if (input.w4var() != '-'):
                if (input.w4mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.w4var(), c = input.w4col(),  s = input.sl1(),label = input.w4var())
                    pushlog2(f"... extra plot #4 variable= {input.w4var()} color = {input.w4col()} NOTE: missing values are dropped.")
                    #ax.add(sb.Dot())    
                elif (input.w4mark() == 'line'):
                    pushlog2(f"... extra plot #4  variable= {input.w4var()} color = {input.w4col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.w4var(), color = input.w4col(),label = input.w4var())
                    #plt.legend(title = input.w4var(), loc = 1, fontsize = 12)
            if (input.w5var() != '-'):
                if (input.w5mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.w5var(), c = input.w5col(),  s = input.sl1(),label = input.w5var())
                    pushlog2(f"... extra plot #5 variable= {input.w5var()} color = {input.w5col()} NOTE: missing values are dropped.")  
                elif (input.w5mark() == 'line'):
                    pushlog2(f"... extra plot #5 variable= {input.w5var()} color = {input.w5col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.w5var(), color = input.w5col(),label = input.w5var())
                    #plt.legend(title = input.w5var(), loc = 1, fontsize = 12)
            if (input.w6var() != '-'):
                if (input.w6mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.w6var(), c = input.w6col(),  s = input.sl1(),label = input.w6var())
                    pushlog2(f"... extra plot #6 variable= {input.w6var()} color = {input.w6col()} NOTE: missing values are dropped.")  
                elif (input.w6mark() == 'line'):
                    pushlog2(f"... extra plot #6 variable= {input.w6var()} color = {input.w6col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.w6var(), color = input.w6col(),label = input.w6var())
            plt.legend(loc = 1, fontsize = 12)                    

            ax.set_title(titlestr)
            ax.set_xlabel(xv)
            ax.set_ylabel(yv)
            ax.set_xlim(input.xlb()-squidgeX,input.xub()+squidgeX)
            ax.set_ylim(input.ylb()-squidgeY,input.yub()+squidgeY)
            if (cv != None) :
                ax.legend(title = cv, handles = lpatches)
        elif (xv != '-'):
            if (input.rb1() == 'Boxplot'):
                ax =sb.boxplot(data = dfg, y=xv,x=color_data, notch = True)
                #ax =sb.boxplot(data = dfg, y=xv,x=colorlist, notch = True)
                ax.set_title(titlestr)
            else: 
               if (input.rb1() == "Kernel Density"):
                   kdeflag = True
               if (cv != None):
                   edgecol = None
               else: edgecol = 'black'
               ax = sb.histplot(data = dfg, x=xv, hue = color_data, ec = edgecol ,kde = kdeflag,bins=input.sl1())
               #ax = sb.histplot(data = dfg, x=xv, color = colorlist, ec = edgecol,kde = kdeflag,bins=input.sl1())
               ax.set_title(titlestr)

               #ax.set_title(f'File: {input.file1()[0]["name"]} #rows = {nrow}')
               ax.set_xlabel(xv)
               ax.set_xlim(input.xlb()-squidgeX,input.xub()+squidgeX)
        else: 
            return
        return plt.draw()

    @render.download(filename="plotIt_Plotting_data.csv")
    def downloadDP():
        df = plt_data()
        print("Download plotting data to: plotIt_Plotting_data.csv")
        #create the row subset for graphing    
        for item in list(subdict().keys()):
            df = df[df[item].astype('str').isin(list(subdict()[item]))]
        for item in list(rngdict().keys()):
            df = df[(df[item]>=rngdict()[item][0]) & (df[item]<=rngdict()[item][1])]
        yield df.to_csv(index = False)

##########################################################################
####  Plotting Tools panel
##########################################################################
    #event observer to update subsetting dictionary
    @reactive.effect
    @reactive.event(input.fvar)
    def newfilter():
        df = plt_data()
        pushlog2("newfilter()")
        if len(df) == 0: return
        #if fvar is not set, restore all rows
        if (input.fvar() == '-'): 
            pushlog2("...Resetting row filter, all rows active.")
            #fct used for subsetting (fct short for factor)
            cols = list(df.columns)
            num_var = list(df.select_dtypes(include=np.number).columns)
            fct_var = [item for item in cols if ((item not in num_var) or (len(list(df[item].unique())) <= max_factor_values))]
            #fctc_var = [item for item in fct_var if (len(list(df[item].unique()))<=5)]#10
            fct_var.insert(0,"-")
            #fctc_var.insert(0,"-")
            newdict = {}
            newdict = {item: list(map(str,list(df[item].unique()))) for item in fct_var if item != '-'}
            subdict.set(newdict)
            ui.update_selectize("fitems",choices = [], selected = [])
            return
        #fvar is set, update filtering variable and set filter items (choices)
        fv = input.fvar()
        inc_items = list(df[fv].astype('str').unique())
        ui.update_selectize("fitems", choices = inc_items, selected = inc_items)

    @reactive.effect
    @reactive.event(input.fitems)
    def subdict_update():
        pushlog2("subdict_update()")
        #update the dictionary of currently active rows keys=col names values = lists of active row values
        fv = input.fvar()
        if (fv == '-'): return
        newdict = subdict()
        newdict[fv] = list(input.fitems())
        subdict.set(newdict)
        pushlog(f"...Plot dictionary update:  Var = {fv}; Active values: {', '.join(newdict[fv])}")

    @reactive.effect
    @reactive.event(input.rfvar)
    def clear_rangefilter():
        if plt_data().empty: return
        df = plt_data()
        num_var = list(df.select_dtypes(include=np.number).columns)
        if input.rfvar() == '-':
            #clear all range filters
            newRdict = {}
            newRdict = {item: (df[item].min(),df[item].max()) for item in num_var}
            rngdict.set(newRdict)
            ui.update_numeric("rflo", value = 0)
            ui.update_numeric("rfhi", value = 0)
        else:
            currdict = rngdict()
            ui.update_numeric("rflo", value = currdict[input.rfvar()][0])
            ui.update_numeric("rfhi", value = currdict[input.rfvar()][1])
        return
    
    @reactive.effect    
    @reactive.event(input.rflo, input.rfhi)
    def do_ranges():
        if plt_data().empty: return
        if input.rfvar() == '-': return
        df = plt_data()
        num_var = list(df.select_dtypes(include=np.number).columns)
        if input.rfvar() == '-': 
            #clear all range filters
            newRdict = {}
            newRdict = {item: [df[item].min(),df[item].max()] for item in num_var}
            rngdict.set(newRdict)
            print('\n'.join([f'{item}: {rngdict()[item]}' for item in rngdict().keys()]))
            ui.update_numeric("rflo", value = 0)
            ui.update_numeric("rfhi",value = 0)
            return
        df = plt_data()
        curdict = rngdict()
        curdict[input.rfvar()] = [input.rflo(), input.rfhi()]
        #curdict[input.rfvar()][1] = input.rfhi()
        ui.update_numeric("rflo", value = curdict[input.rfvar()][0])
        ui.update_numeric("rfhi", value = curdict[input.rfvar()][1])

        rngdict.set(curdict)
        pushlog(f"updating data ranges: column = {input.rfvar()}, entry: [{rngdict()[input.rfvar()][0]},{rngdict()[input.rfvar()][1]}]")
        
        #ui.update_numeric("xlb",value = min(df[input.xvar()]))
        #ui.update_numeric("xub",value = max(df[input.xvar()]))
    
    # @reactive.effect
    # @reactive.event(input.rfvar)
    # def do_refvar():

    #displays log of currently active rows
    @render.text
    @reactive.event(input.updateB,input.fvar,input.fitems)
    def log(): 
        if (plt_data().empty): return
        pushlog2("log(): show active rows")
        pushlog2('\n'.join([f'{item}: {subdict()[item]}' for item in subdict().keys()]))
        return '\n'.join([f'{item}: {subdict()[item]}' for item in subdict().keys()])
        
    @render.text
    @reactive.event(input.updateB,input.rfvar,input.rflo, input.rfhi)
    def rlog(): 
        if (plt_data().empty): return
        pushlog2("rlog(): current ranges")
        pushlog2('\n'.join([f'{item}: {rngdict()[item]}' for item in rngdict().keys()]))
        #print("rlog: show current row ranges.")
        return '\n'.join([f'{item}: {rngdict()[item]}' for item in rngdict().keys()])
        
##########################################################################
####  Linear Models panel
##########################################################################
    @render.ui
    @reactive.event(input.mtype)
    def nbparm():
        if ("NEGATIVE_BINOMIAL" == input.mtype()):
            return ui.TagList(*[ ui.input_numeric("alphaparm", "Negative Binomial: variance parm (between 0.01 and 2.0):", value="1.0",width=10)])
        else:
            return ui.TagList(*[])

    @reactive.effect
    @reactive.event(input.depvar)    
    def do_depvar():
        df = plt_data()        #mdl.set(None)
        if len(df) == 0: return
        num_var = list(df.select_dtypes(include=np.number).columns)
        str_var = list(df.select_dtypes(exclude=np.number).columns)
        if (input.depvar() == '-'): 
            ui.update_selectize('depvar',choices = ['-'] + num_var)
            ui.update_selectize('indvar',choices = num_var + str_var)
        else: 
            indvar_choices = [item for item in num_var+str_var if item != input.depvar()]
            ui.update_selectize('indvar',choices = indvar_choices)
        return
                        
    @reactive.effect
    @reactive.event(input.indvar,input.depvar)
    def doMstring():
        ui.update_text("stringM",value = f"{''.join(input.depvar())} ~ {' + '.join(input.indvar())}")
        return
    
    @reactive.effect
    @reactive.event(input.modelGo)
    def runModel():
        df = plt_data()
        pushlog(f"At runModel Top.::::::  input.indvar = {input.indvar()}")
        size0 = len(df)
        if (input.depvar() == '-'): 
            return
        pushlog(f"runModel:: Dependent Var: {input.depvar()}, Independent Var: {', '.join(input.indvar())}, model: {input.stringM()}")
        #apply the current subset items are column names items are the dictionary keys, 
        #   the dictionary entry is the list of active row values for that column.
        for item in list(subdict().keys()):
            df = df[df[item].astype('str').isin(list(subdict()[item]))]
        size1 = len(df)
        #  apply the row range filters
        for item in list(rngdict().keys()):
            df = df[(df[item]>=rngdict()[item][0]) & (df[item]<=rngdict()[item][1])]
        size1b = len(df)
        #create factors for numerical variables as set by user by changing them into strings
        if len(input.tofactor()) >0 :
            pushlog(f"runModel: Creating factors for: {', '.join(input.tofactor())}")
        for item in input.tofactor():
            df[item] = df[item].astype('str')    
        #manually remove rows containing NaNs in the dependent or independent variables columns
        df.dropna(subset = [input.depvar()] + list(input.indvar()),inplace = True)   
        size2 = len(df)   
        #check to see if the dependent variable is binary (use logit) has several outcomes (use ols) or just one (quit)
        pushlog(f".Rows read in: {size0},... {size0-size1} rows deleted by categorical filter, \n.... {size1-size1b} deleted by range filter, \n.... {size1b-size2} deleted due to missing data.")        
        #minimal sanity check: a) dependent variable can't be constant b) if LOGIT has been chosen, dependent variable must be binary (0 or 1)
        outcomes = list(df[input.depvar()].unique())
        #ISINT = False
        #if df[input.depvar()].dtype == 'int':
        #    ISINT = True
        #elif df[input.depvar()].dtype == 'float':
        #    if (df[input.depvar()].apply(float.is_integer).all()):
        #        ISINT = True
        #    else:
        #        ISINT = False
        #else: 
        #    ISINT = False
        pushlog(f"...Dependent variable type: {df[input.depvar()].dtype}")
        no_outcomes = len(outcomes)
        if (no_outcomes <=1):
            pushlog("The dependent variable is a constant.  I'm confused.  Please check and try again.")
            mdl.set(None)
            return
        STOP = False
        if (input.mtype() == 'LOGIT'):
            if set([0,1]) == set(outcomes) :
                try:
                    #res = smf.logit(formula = input.stringM(), data=df).fit()
                    res = smf.glm(formula = input.stringM(), data = df, family=sm.families.Binomial()).fit()
                except: 
                    STOP = True
                mdl_type.set('LOGIT')
            else:
                pushlog("...Logistic Regression Error: dependent variable not binary 0,1.")
                mdl.set(None)
                return
        elif (input.mtype() == 'OLS'):
            try:
                res = smf.ols(formula=input.stringM(), data=df).fit()
            except: 
                STOP = True            
            mdl_type.set('OLS')
        elif (input.mtype() == 'POISSON') & (min(outcomes) >=0):
            if (min(outcomes) >= 0) & ISINT:
                try:
                    pushlog(f" ...Estimating model: ISINT= {ISINT}, min outcome = {min(outcomes)}")
                    res = smf.glm(formula=input.stringM(), data=df, family = sm.families.Poisson()).fit()
                except: 
                    STOP = True            
                mdl_type.set('POISSON')
            else:
                pushlog("...Poisson Regression Error: dependent variables are not non-negative integers.")
                mdl.set(None)
                return
        elif (input.mtype() == 'NEGATIVE_BINOMIAL') & (min(outcomes) >=0):
            if (min(outcomes) >= 0) & ISINT:
                try:
                    pushlog(f" ...Estimating model: ISINT= {ISINT}, min outcome = {min(outcomes)}")
                    res = smf.glm(formula=input.stringM(), data=df, family = sm.families.NegativeBinomial(alpha = input.alphaparm())).fit()
                except: 
                    STOP = True            
                mdl_type.set('NEGATIVE_BINOMIAL')
            else:
                pushlog("...Negative Binomial Regression Error: dependent variables are not non-negative integers.")
                mdl.set(None)
                return
        else:
            STOP = True
            pushlog("No model chosen. Choose OLS, LOGIT, POISSON, or NEGATIVE BINOMIAL")
        if STOP:
            mdl_indvar.set([])
            mdl_depvar.set('-')
            mdl.set(None)
            ui.update_radio_buttons("datachoose",choices = ['Input Data','Model Data'],selected = None)
            pushlog(f"{mdl_type()} estimation failed. Model = {input.stringM()} ")
            return
        # regression succeeded          
        mdl.set(res) 
        mdl_d = pd.concat([res.model.data.orig_exog,res.model.data.orig_endog],axis = 1)
        
        if (mdl_type() != 'OLS') :
            prediction_res = res.get_prediction(transform = True)
            res_frame= prediction_res.summary_frame(alpha = input.siglev())
            mdl_d['CI_lb'] =  res_frame['mean_ci_lower']
            mdl_d['CI_ub'] =  res_frame['mean_ci_upper']
            #mdl_d['PI_lb'] =  res_frame['obs_ci_lower']
            #mdl_d['PI_ub'] =  res_frame['obs_ci_upper']
            mdl_d['Predictions'] = res_frame['mean']
            mdl_d['Deviance_Resid'] = res.resid_deviance
        elif (mdl_type()=='OLS'):
            prediction_res = res.get_prediction(transform  = True)
            res_frame= prediction_res.summary_frame(alpha = input.siglev())
            mdl_d['CI_lb'] =  res_frame['mean_ci_lower']
            mdl_d['CI_ub'] =  res_frame['mean_ci_upper']
            mdl_d['PI_lb'] =  res_frame['obs_ci_lower']
            mdl_d['PI_ub'] =  res_frame['obs_ci_upper']
            mdl_d['Predictions'] = res_frame['mean']
            mdl_d['Residuals'] =  res.resid
        addoncols = [item for item in df.columns if item not in mdl_d.columns]
        #reconvert numeric variables that were turned into strings (and subsequently factors)
        for item in input.tofactor():
            df[item] = pd.to_numeric(df[item])    
        #prepare the data set for applying the plotting tools to the model
        pushlog(f"...Model data columns: {','.join(mdl_d.columns)}")
        pushlog(f"...Extra data columns: {','.join(addoncols)}")
        mdl_d = pd.concat([mdl_d,df[addoncols]],axis = 1)
        mdl_data.set(mdl_d)
        mdl_indvar.set(input.indvar())
        mdl_depvar.set(input.depvar())
        mdl_stringM.set(input.stringM())
        #now setup the standard plots
        ui.update_select("lmsp",choices = ['-'] + list(res.params.index),selected = None)
        if (mdl_type() == 'LOGIT'):
            ui.update_radio_buttons("regplotR",choices = ['ROC', 'Partial Regression','Fit'], selected = 'ROC')
        elif(mdl_type() == 'OLS'):
            if (len(input.depvar()) < 12000):
                ui.update_radio_buttons("regplotR",choices = ['Leverage','Partial Regression','Influence','Fit'],selected = 'Leverage')
            else:
                ui.update_radio_buttons("regplotR",choices = ['Partial Regression','Fit'],selected = 'Partial Regression')
        else:
                ui.update_radio_buttons("regplotR",choices = ['Partial Regression','Fit'],selected = 'Partial Regression')

        #set up data choice (input data or mode data) on the plotting tab

        ######Kludge to simulate the user choosing 'Model Data' post model estimation
        with reactive.isolate(): #first we change the 'datachoose" choice to 'None' but we hide it
            ui.update_radio_buttons("datachoose",choices = ['Input Data', 'Model Data'],selected = None)
        #then we choose 'Model Data' but we don't hide it -- this triggers dataUpdate() and setupPlot() to initialize the model data as the  plotting data.
        ui.update_radio_buttons("datachoose",label = "Data: (selecting Input Data clears model)",choices = ['Input Data','Model Data'],selected = 'Model Data')
        
        pushlog(f"Estimation successful. {mdl_type()} ...Model: {mdl_stringM()}")
        #pushlog(str( mdl().summary()))
        return
    
    @render.text
    @reactive.event(input.modelGo,mdl)
    def modelSummary():
        if (mdl() == None) :
            outstring = f"Model estimation failed or model cleared.  Check log for details. model: {input.stringM()}"
            return outstring
        SSstr0 =  "=============================================================================="
        if mdl_type() == 'OLS' :
            #SSstr1 =  f"SSE = {round(mdl().ssr,5)}, SSR={round(mdl().ess,5)}, SST = {round(mdl().ssr + mdl().ess,5)}" 
            #SSstr2 = f"MSE = {round(mdl().mse_resid,5)}, MSR ={round(mdl().mse_model,5)}, MST = {round(mdl().mse_total,5)} "   
            SSstr = SSstr0 + "\n" + "Model: " + input.stringM() + "\n\n" + str( mdl().summary()) + "\n"
            SSstr = SSstr + SSstr0 + '\n' + 'Analysis of Variance' + "\n"
            anova_rep = sm.stats.anova_lm(mdl(),typ=1)          
            row1 = pd.Series({'df': mdl().df_model, 'sum_sq': mdl().ess,'mean_sq': mdl().mse_model,'F': ' ','PR(>F)':' '},name = 'Regression')
            row2 = pd.Series({'df': mdl().df_resid + mdl().df_model, 'sum_sq': mdl().ess + mdl().ssr,'mean_sq': mdl().mse_total, 'F': ' ','PR(>F)':' '},name = 'Total')
            anova_rep.loc['Regression'] = row1
            anova_rep.loc['Total'] = row2
            anova_rep.replace(np.nan," ")
            SSstr = SSstr + str(anova_rep[anova_rep.columns[0:len(anova_rep.columns)-2]][-3:]) + '\n' + SSstr0
        elif mdl_type() == 'LOGIT' :
            fpr, tpr, thresholds = roc_curve(mdl_data()[input.depvar()], mdl_data()['Predictions']) 
            roc_auc = auc(fpr, tpr)  
            SSstr1 = f" AUC: {roc_auc}"
            SSstr = "Model: " + input.stringM() + "\n\n" + str( mdl().summary()) + "\n" + SSstr0 + "\n" + SSstr1 + "\n" + SSstr0
        else: SSstr = "Model: " + input.stringM() + "\n\n" + str( mdl().summary()) 
        pushlog(SSstr)
        return SSstr
        
    @render.ui
    @reactive.event(input.modelGo)
    def dloads():
        if (mdl() != None):
            return ui.TagList(
               ui.column(3,offset = 0,*[ui.download_button("downloadD", "Download Data Set")]),
               )
        else:
            return None
              
    @render.download(filename="plotIt_Model_Data.csv")
    def downloadD():
        yield mdl_data().to_csv(index = False)
         
##########################################################################
####  Linear Models: Standard Plots panel
##########################################################################       
    @render.plot
    #@reactive.event(mdl)
    def regplot1():
        if (mdl() == None) : 
            #lm_msgstr.set("Run a model using the Linear Model tab before plotting it (first numbers, then pictures)")
            return
        #lm_msgstr.set("")
        fig = plt.figure(figsize=(8, 8)) 
        if (input.regplotR() == 'Partial Regression'):
            plot_partregress_grid(mdl(), fig=fig)
            plt.axhline(y=0)
            return plt.draw()
        if (input.regplotR() == 'ROC') :
            if (mdl_type() != 'LOGIT') : return
            # Calculate ROC curve
            if (input.depvar() == '-') :
                pushlog(f"Unable to display ROC curve: dependent variable not specified: {input.depvar()}")
                return
            fpr, tpr, thresholds = roc_curve(mdl_data()[input.depvar()], mdl_data()['Predictions']) 
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve
            ax = fig.add_subplot()
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f"ROC  Model: {input.stringM()}, AUC={round(roc_auc,5)}")
            return plt.draw()
        if (input.regplotR() == 'Leverage'):
            if(mdl_type()!= 'OLS'): return
            plot_leverage_resid2(mdl())          
            return plt.draw()
        if (input.regplotR() == 'Influence'):
            if mdl_type() != 'OLS' : return
            influence_plot(mdl(),fig = fig)          
            return plt.draw()
        if (input.regplotR() == 'Fit'):
            ivs = list(mdl().params.index)
            targetv = input.lmsp()
            if (targetv in ivs):
                varno = ivs.index(targetv)  
                fig = plot_fit(mdl(),varno,vlines = False)
            else:
                return
            fig.tight_layout(pad=1.0)
            return plt.draw()

#app = App(app_ui, server,debug=True)
app = App(app_ui, server)

