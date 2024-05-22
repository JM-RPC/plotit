#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:45:31 2024

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
import plotly.express as pltx
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import shinywidgets
from shinywidgets import render_widget, output_widget
import plotly.express as pltx
import plotly.graph_objs as go
import os
import signal
from datetime import datetime
from shinywidgets import output_widget, render_widget

#import nest_asyncio


#from shinywidgets import output_widget, render_widget

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
    ui.nav_panel("Input",
        ui.input_file("file1", "Choose .csv or .dta File", accept=[".csv",".CSV",".dta",".DTA"], multiple=False, placeholder = ''),
        ui.output_text('io_mess'),ui.input_radio_buttons('killna', 'Remove rows with missing data in one or more columns?',choices = ['No','Yes']),
        ui.output_text_verbatim("info"), 
        ui.output_table("summary"),
        ),
    ui.nav_panel("Correlations",
                 ui.row(ui.input_selectize("corrV","Select variables:",choices = [''],multiple = True,width = "200px")),
                 ui.output_plot("dataPD",width = '1200px', height = '1200px'),
                 ),
    ui.nav_panel("Plotting",
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
                 ui.row(
                     ui.input_selectize("fvar","Filter On:" ,choices = ['-'], multiple=False),
                     ui.input_selectize("fitems","Included Rows:",choices = ['-'], multiple=True),
                     ),
                 ui.row(
                        ui.HTML("<p>Rows Selected (filter on \"-\" above to clear filter).</p>"),
                     ),
                 ui.row(
                     ui.output_text_verbatim("log")
                     ),
                 ui.row(
                        ui.input_text("titleplt","Plot Title:", value = '-', width = '800px', )
                     ),
                 ui.row(ui.HTML("<p>These bounds override plot bounds.  To reset reselect the variable.</p>")
                     ),
                 ui.row(
                     ui.column(2,offset=0,*[ ui.input_numeric("xlb", "X lower bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("xub", "X upper bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("ylb", "Y lower bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("yub", "Y upper bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("zlb", "Z lower bound:", value="",width=10)]),
                     ui.column(2,offset=0,*[ ui.input_numeric("zub", "Z upper bound:", value="",width=10)]),
                     ),
                 ),
    ui.nav_panel("Plot Extras",
                   ui.row(ui.HTML("<p> Additional data series for 2D and 3D plots</p>"),
                      ),
                   ui.row(
                       ui.column(2,offset=0,*[ ui.input_selectize("y1var","Y1 variable (2D):",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("y2var","Y2 variable (2D):",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("y3var","Y3 variable (2D):",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("z1var","Z1 variable (3D):",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("z2var","Z2 variable (3D):",choices = ['-'], multiple=False)]),
                       ui.column(2,offset=0,*[ ui.input_selectize("z3var","Z3 variable (3D):",choices = ['-'], multiple=False)]),
                       ),
                    ui.row(
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("y1mark","Type:",choices = ['dot','line'],inline = True)]),                    
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("y2mark","Type:",choices = ['dot','line'],inline = True)]),
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("y3mark","Type:",choices = ['dot','line'],inline = True)]),
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("z1mark","Type:",choices = ['dot','line'],inline = True)]),
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("z2mark","Type:",choices = ['dot','line'],inline = True)]),
                        ui.column(2,offset=0,*[ ui.input_radio_buttons("z3mark","Type:",choices = ['dot','line'],inline = True)]),
                        ),
                   ui.row(
                       ui.column(2,offset=0,*[ ui.input_selectize("y1col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("y2col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("y3col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("z1col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("z2col","Color:",choices = basecolors0, multiple = False)]),                    
                       ui.column(2,offset=0,*[ ui.input_selectize("z3col","Color:",choices = basecolors0, multiple = False)]),                    
                       ),
                   ui.row(
                       ui.column(2,offset=0,*[ ui.input_numeric("siglev", "Significance Level:", value="0.05",width=20)]),
                       ),
                   ),
    ui.nav_panel("Linear Models",
                 ui.row(
                     ui.column(6,offset=0,*[ui.input_radio_buttons("mtype","Model Type",choices = ['OLS','LOGIT'],inline = True)]),
                     ),
                 ui.row(
                     ui.column(3, offset = 0, *[ui.input_selectize("depvar","Dependent Variable:",choices = ['-'],multiple = False)]),
                     ui.column(5, offset = 0, *[ui.input_selectize("indvar","Independent Variables:", choices = ['-'],multiple = True)]),
                     ui.column(4, offset = 0, *[ui.input_selectize("tofactor","Convert Numeric Variables to factors:", choices = ['-'],multiple = True)]),
                     ),
                 ui.row(ui.HTML("<p> Use Wilkinson/Patsy notation to specify variable transformations.</p>")
                     ),
                 ui.row(
                     ui.column(9, offset = 0, *[ui.input_text('stringM','Model String:',width = '1000px')]),
                     ui.column(3, offset = 0, *[ui.input_action_button('modelGo',"Run Model")]),
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
                  ui.output_text_verbatim("logout")
                  ),
    ui.nav_spacer(),
    ui.nav_menu("Exit",
                 ui.nav_control(
                     ui.column(1,offset=0,*[ui.input_action_button("exit","Exit")]),
                     ),
                 ),
    
               
underline = True, title = "plotIt v.0.0.3 ")


def server(input: Inputs, output: Outputs, session: Session):
    mdl_type = reactive.value("OLS") #currently supported types: OLS, LOGIT
    mdl = reactive.value(None)
    mdl_depvar = reactive.value('-')
    mdl_indvar = reactive.value(())
    mdl_stringM = reactive.value(" - ~ ")
    mdl_data = reactive.value(pd.DataFrame())
    subdict = reactive.value({})
    logstr = reactive.value("")
    dbgstr = reactive.value(f"At server start: Figures: {plt.get_fignums()} \n")
    plt_msgstr = reactive.value("")
    io_msgstr = reactive.value("")
    lm_msgstr = reactive.value("")
    plt_data = reactive.value(pd.DataFrame())
    logstr = reactive.value(f"Log Start: {datetime.now()}")
   
    @reactive.effect
    @reactive.event(input.exit)
    async def do_exit():
        #plt.close(fig)
        await session.app.stop()
        os.kill(os.getpid(), signal.SIGTERM)
        max_factor_values = 50

#        basecolorsalpha = ['red',  'blue', 'green', 'goldenrod', 'violet','cyan', 'yellow','grey','gold','magenta','silver','orange','olive','khaki','thistle']
#        basecolors = [matplotlib.colors.to_rgba(item,alpha = None) for item in basecolorsalpha]

        #fig = plt.figure(figsize = (10,8),tight_layout=True)
        
##########################################################################
####  Log panel
##########################################################################
    @render.text  
    @reactive.event(input.logGo)
    def logout():
        return logstr() 

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
        else: 
            fpath = str(input.file1()[0]['datapath'])
            if (fpath[-4:] == '.csv') or (fpath[-4:] == '.CSV'):
                df = pd.read_csv(input.file1()[0]["datapath"])
            else:
                df = pd.read_stata(input.file1()[0]["datapath"])
            pushlog("************************************************")
            pushlog("File read: "  + input.file1()[0]['name'])
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
            plt_data.set(df)
            return df

    @render.text
    def info():
        df = parsed_file()
        #df = plt_data()
        if df.empty:
           return 
        #display df.info
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        return s
    
    @render.table
    def summary():
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

    #warnings for data input panel ++ this doesn't seem to work
    # @render.text
    # #@reactive.event(input.updateB)
    # def io_mess():
    #     return io_msgstr()  
    
##########################################################################
####  Correlations panel
##########################################################################
    
    @render.plot
    @reactive.event(input.corrV)
    def dataPD():
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
        with reactive.isolate():
            if (input.datachoose() == 'Model Data') :
                pushlog("Switching to model data.")
                plt_data.set(mdl_data())
            if (input.datachoose() == 'Input Data'):
                pushlog("Switching to original input data.")
                plt_data.set(parsed_file())


    @reactive.effect
    @reactive.event(parsed_file, input.datachoose)
    def setupPlot():
#        print("....In setupPlot")
        pushlog("...Initializing Plotting Data (setupPlot)")
        df = pd.DataFrame()
        df = plt_data()
        nrow = len(df)
        if (nrow == 0): 
#            print(f"setupPlot...{nrow} rows in current data datachoose = {input.datachoose()}")
            return
        cols = list(df.columns)
        num_var = list(df.select_dtypes(include=np.number).columns)

        str_var = [item for item in cols if item not in num_var]    

        #fct used for subsetting (fct short for factor) and coloring

        fct_var = [item for item in cols if ((item not in num_var) or (len(list(df[item].unique()))<=max_factor_values))]
        #subset dictionary
        newdict = {}                
        newdict = {item: list(map(str,list(df[item].unique()))) for item in fct_var}
        subdict.set(newdict)

        num_fct = [item for item in list(df.columns) if (item in num_var) and len(list(df[item].unique())) <= max_factor_values]
        ui.update_selectize("xvar",choices = ['-']+num_var)
        ui.update_selectize("yvar",choices = ['-']+num_var)
        ui.update_selectize("zvar",choices = ['-']+num_var)
        ui.update_selectize("cvar",choices = ['-']+fct_var)
        ui.update_selectize("fvar",choices = ['-']+fct_var)
        ui.update_selectize("corrV", choices = num_var, selected = None)
        if (input.datachoose() != 'Model Data'):
          ui.update_selectize("indvar",choices = num_var + str_var)
          ui.update_selectize("depvar",choices =  ['-'] + num_var)
        ui.update_selectize("tofactor",choices =  num_fct) 
        ui.update_selectize("y1var",choices = ['-'] + num_var)        
        ui.update_selectize("y2var",choices = ['-'] + num_var)        
        ui.update_selectize("y3var",choices = ['-'] + num_var)        
        ui.update_selectize("z1var",choices = ['-'] + num_var)        
        ui.update_selectize("z2var",choices = ['-'] + num_var)        
        ui.update_selectize("z3var",choices = ['-'] + num_var)        
        return
    
    #event observer to update subsetting dictionary
    @reactive.effect
    @reactive.event(input.fvar)
    def newfilter():
        df = plt_data()
        if len(df) == 0: return
        #if fvar is not set, restore all rows
        if (input.fvar() == '-'): 
            pushlog("Resetting row filter, all rows active.")
            #fct used for subsetting (fct short for factor)
            cols = list(df.columns)
            num_var = list(df.select_dtypes(include=np.number).columns)
            fct_var = [item for item in cols if ((item not in num_var) or (len(list(df[item].unique()))<=max_factor_values))]
            #fctc_var = [item for item in fct_var if (len(list(df[item].unique()))<=5)]#10
            fct_var.insert(0,"-")
            #fctc_var.insert(0,"-")
            newdict = {}
            newdict = {item: list(map(str,list(df[item].unique()))) for item in fct_var if item != '-'}

            #for item in fct_var:
            #    if item != '-' : newdict[item] = list(map(str,list(df[item].unique())))
            
            subdict.set(newdict)
            ui.update_selectize("fitems",choices = [], selected = [])
            return
        fv = input.fvar()
        inc_items = list(df[fv].astype('str').unique())
        ui.update_selectize("fitems", choices = inc_items, selected = inc_items)

    @reactive.effect
    @reactive.event(input.fitems)
    def subdict_update():
        #update the dictionary of currently active rows keys=col names values = lists of active row values
        fv = input.fvar()
        if (fv == '-'): return
        newdict = subdict()
        newdict[fv] = list(input.fitems())
        subdict.set(newdict)
        pushlog(f"Plot dictionary update:  Var = {fv}; Active values: {', '.join(newdict[fv])}")
        
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
        
    #displays log of currently active rows
    @render.text
    @reactive.event(input.updateB,input.fvar,input.xvar, input.yvar, input.zvar, input.cvar)
    def log():  
        if 1==1: #input.fvar() != '-':
            return '\n'.join([f'{item}: {subdict()[item]}' for item in subdict().keys()])
        else:
            return ""
        
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
            return ui.TagList(
                              #ui.column(1,offset = 0,*[ui.input_action_button("updateB3", "Update")]),
                              ui.column(3,offset=0,*[ui.input_slider("sl1","# dotsize",min = 0, max = 40, value = 10)]),
                              ui.column(1,offset = 0,),
                              ui.column(4,offset=0,*[ui.input_checkbox_group("scttropts","3D Scatter Plot Options:",
                                                choices=('Show Trend','CI','PI'),selected=(),inline = True)])
                              )
        elif ((input.yvar() != '-') & (input.xvar() != '-')):           
            return ui.TagList(
                              #ui.column(1,offset = 0,*[ui.input_action_button("updateB12", "Update")]),
                              ui.column(3,offset=0,*[ui.input_slider("sl1","# dotsize",min = 0, max = 40, value = 10)]),
                              ui.column(1,offset = 0,),
                              ui.column(2,offset=0,*[ui.input_checkbox_group("scttropts","Scatter Plot Options:",
                                                                    choices=('Show Trend','CI','PI'),selected=(),inline = True)])
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
        
    @output
    @render_widget
    @reactive.event(input.updateB)
    def plot3():
        plt.clf()
        if (input.xvar() == '-') | (input.yvar() == '-') | (input.zvar() == '-'): return
        df = plt_data()
        if len(df) == 0: 
#            plt_msgstr.set("You need a data set before you can plot.")
            return
        xv = input.xvar()
        yv = input.yvar()
        zv = input.zvar()
        z1v = input.z1var()
        z2v = input.z2var()
        z3v = input.z3var()
        totrow = len(df)
        cv = input.cvar()
        #pushlog(f"Plotting: x = {xv}, y={yv}, z={zv}, color = {cv} ")
        if cv == '-': cv = None 
         #take out the rows that the user has decided to ignore
        for item in list(subdict().keys()):
            df = df[df[item].astype('str').isin(list(subdict()[item]))]
        nrow0 = len(df)
        dfg = df.dropna(subset = [xv, yv, zv]) #get rid fo rows with na's in the columns to be plotted
        nrow = len(dfg)
        #plotly can only plot up to about 400,000 to 450,000 rows of data, randomly downsample if needed.  Warn user in the title & log
        if nrow > 400000:
            dfg = dfg.sample(400000)
            nrow = len(dfg)
            ttlstr = f"File: {input.file1()[0]['name']} <br> down-sampled!  {nrow} rows plotted out of {totrow} "
            pushlog("Plotly 3D plotting limit warning! Attempted to plot {nrow} rows, downsampled to 400,000 rows.")
        else:
            ttlstr = f"File: {input.file1()[0]['name']}:  {nrow} rows plotted out of {totrow} "
            pushlog(f"plot3: {totrow-nrow0} rows filtered out, {nrow0-nrow} rows removed due to missing data.")
        if input.titleplt() != "-" :
            ttlstr = input.titleplt()
        if (cv != None):
            nucolor ,nupatches, colorD = getcolor(list(dfg[cv].astype('str'))) #fix tup color map
            colormap = {str(item) : f"rgb({int(250*float(colorD[item][0]))},{int(250*float(colorD[item][1]))},{int(250*float(colorD[item][2]))})" for item in colorD.keys() }
            fig = pltx.scatter_3d(dfg,x=xv, y= yv, z = zv, color = list(dfg[cv].astype('str')), color_discrete_map = colormap, width = 900, height = 600, title = ttlstr)
        else:
            fig = pltx.scatter_3d(dfg,x=xv, y= yv, z = zv, width = 900, height = 900, title = ttlstr)
        fig.update_traces(marker = dict(size = int(input.sl1()/5 +1))) # fix up dot size
        if  "Show Trend" in input.scttropts(): 
            res = None
            MTYPE = None
            #if the x, y and z variables perfectly match the variables in the most recently estimated model, then use that model's results 
            #otherwise fit a new model
            if ((input.datachoose() == "Model Data") & (len(mdl_indvar())==2) & (xv in mdl_indvar()) & (yv in mdl_indvar()) 
                & (zv == mdl_depvar()) & (mdl() !=  None)):
                pushlog(f"plot3: using extant model....string={mdl_stringM()}; indvar={mdl_indvar()}; depvar = {mdl_depvar()}")
                res = mdl()
                MTYPE = mdl_type()
            else:# otherwise fit z against x and y from scratch us logit if z is binary (0,1)
                pushlog(f"plot3: fitting response surface model....string={mdl_stringM()}; indvar={mdl_indvar()}, depvar = {mdl_depvar()}")
                if (set([0,1]) == set(df[zv])):
                    try:
                        res = smf.glm(formula = f"{zv} ~ {xv} + {yv}" , data = dfg, family=sm.families.Binomial()).fit()
                        #res = smf.logit(f"{zv} ~ {xv} + {yv}" ,data= dfg).fit()
                        MTYPE = 'LOGIT'
                    except:
                        res = None
                else:
                    try:
                        res = smf.ols(f"{zv} ~ {xv} + {yv}" ,data= dfg).fit()
                        MTYPE = 'OLS'
                    except:
                        res = None
            sq = 0.05               
            deltax = dfg[xv].max() - dfg[xv].min()
            deltay = dfg[yv].max() - dfg[yv].min()
            xlo = dfg[xv].min() - sq*deltax
            xup = dfg[xv].max() + sq*deltax
            ylo = dfg[yv].min() - sq*deltay
            yup = dfg[yv].max() + sq*deltay
            gridcount = 25                          
            # create data for the response surface 
            if (res != None):             
                xvars, yvars = np.meshgrid(np.arange(xlo,xup,deltax/gridcount),
                                 np.arange(ylo, yup,deltay/gridcount))                
                exog0 = pd.DataFrame({xv: xvars.ravel(), yv: yvars.ravel()}) 
                #calculate values of the dependent variable (z) for the response surface 
                #res_predictions = res.get_prediction(exog=exog0,transform = True)
                #res_frame = res_predictions.summary_frame(alpha = input.siglev())


                res_predictions = res.get_prediction(exog=exog0,transform = True)
                res_frame = res_predictions.summary_frame(alpha = input.siglev())
                znew = res_frame['mean'].values.reshape(xvars.shape)    
                Ci_lb1 =  res_frame['mean_ci_lower'].values.reshape(xvars.shape)
                Ci_ub1 =  res_frame['mean_ci_upper'].values.reshape(xvars.shape)
                if (MTYPE == 'OLS'):
                    Pi_lb1 =  res_frame['obs_ci_lower'].values.reshape(xvars.shape)
                    Pi_ub1 =  res_frame['obs_ci_upper'].values.reshape(xvars.shape)
                                 
                #we are ready to add the traces
                fig.add_trace(go.Surface(x=xvars,y=yvars,z=znew, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                if ('CI' in input.scttropts()):
                    fig.add_trace(go.Surface(x=xvars,y=yvars,z=Ci_lb1, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                    fig.add_trace(go.Surface(x=xvars,y=yvars,z=Ci_ub1, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                if ('PI' in input.scttropts()) & (MTYPE == 'OLS'):
                    fig.add_trace(go.Surface(x=xvars,y=yvars,z=Pi_lb1, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                    fig.add_trace(go.Surface(x=xvars,y=yvars,z=Pi_ub1, opacity = 0.75,showscale = False)) #dict(orientation = 'h')))
                #fig.update_traces(colorbar = dict(orientation='h', y = -0.25, x = 0.5))
                
        if (z1v != '-'):
            dfg = dfg.dropna(subset = [z1v]) #get rid fo rows with na's in the columns to be plotted
            if (input.z1mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[z1v],marker = dict(size = int(input.sl1()/5 +1))))
                pushlog(f" 3D extra plot #1 variable= {input.z1var()} color = {input.z1col()} NOTE: missing values are dropped.")
            elif (input.z1mark() == 'line'):
                pushlog(f" 3D extra plot #1  variable= {input.z1var()} color = {input.z1col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[z1v]))
        if (z2v != '-'):
            dfg = dfg.dropna(subset = [z2v]) #get rid fo rows with na's in the columns to be plotted
            if (input.z2mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[z2v],marker = dict(size = int(input.sl1()/5 +1))))
                pushlog(f" 3D extra plot #2 variable= {input.z2var()} color = {input.z2col()} NOTE: missing values are dropped.")
            elif (input.z2mark() == 'line'):
                pushlog(f" 3D extra plot #2  variable= {input.z2var()} color = {input.z2col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[z2v]))
        if (z3v != '-'):
            dfg = dfg.dropna(subset = [z3v]) #get rid fo rows with na's in the columns to be plotted
            if (input.z3mark() == 'dot'):
                fig.add_trace(go.Scatter3d(x = dfg[xv], y = dfg[yv], z = dfg[z3v],marker = dict(size = int(input.sl1()/5 +1))))
                pushlog(f" 3D extra plot #2 variable= {input.z3var()} color = {input.z3col()} NOTE: missing values are dropped.")
            elif (input.z3mark() == 'line'):
                pushlog(f" 3D extra plot #2  variable= {input.z3var()} color = {input.z3col()} NOTE: missing values are dropped.")
                fig.add_trace(go.Mesh3d( x = dfg[xv], y = dfg[yv], z = dfg[z3v]))
        return fig
                
    @render.plot
    @reactive.event(input.updateB)
    def Plots():
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
        pushlog(f"plotting x = {xv}, y = {yv}, color = {cv}")
        #print(f"Beginning of plots ...Model: {mdl_stringM()} indvar = {mdl_indvar()}  depvar = {mdl_depvar()}")


        #expand the plot axes a squidge for esthetics
        squidgeVal = 0.05
        if (xv != '-'):
            squidgeX = (input.xub()-input.xlb())*squidgeVal
        if (yv != '-'):
            squidgeY = (input.yub()-input.ylb())*squidgeVal
#        if (zv != '-'):
#            squidgeZ = (input.zub()-input.zlb())*squidgeVal
     
        
        #create the row subset for plotting
      
        for item in list(subdict().keys()):
           df = df[df[item].astype('str').isin(list(subdict()[item]))]
        nrow1 = len(df)
        pushlog(f"{totrow-nrow1} rows excluded via filter.")
        
        #drop rows that have NaN values in the columns to be plotted
        sbst = [xv]
        if yv != '-': sbst.append(yv)
        if zv != '-': sbst.append(zv)
        dfg = df.dropna(subset = sbst)
        nrow = len(dfg)
        pushlog(f"{nrow1 - nrow} rows dropped from plotting set due to missing data.")
        
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
            ax=sb.scatterplot(data = dfg,x = xv,y = yv, c = colorlist,  s = input.sl1())
            #print(f".....Just before Show Trend in plots mdl_indvar = {list(mdl_indvar())}, model: {mdl_stringM()}, input.indvar = {list(input.indvar())}, input.depvar = {input.depvar()}")
            if "Show Trend" in input.scttropts():
                res = None
                if ((input.datachoose() == "Model Data") & (len(mdl_indvar())==1) & (xv in mdl_indvar()) & (yv in mdl_depvar()) 
                       & (mdl() !=  None)):
                    pushlog(f"...in plots using extant model....string={mdl_stringM()} imdl_ndvar={mdl_indvar()} mdl_depvar= {mdl_depvar()} mdl good? {mdl() != None}")
                    res = mdl()
                    MTYPE = mdl_type()
                else:
                    if (set([0,1]) == set(df[yv])):
                        try:
                            res = smf.glm(f"{yv} ~ {xv}" ,data= dfg, family=sm.families.Binomial()).fit()
                            MTYPE = 'LOGIT'
                        except:
                            res = None
                    else:
                        try:
                            pushlog(f"...in plots using new model....string={mdl_stringM()} imdl_ndvar={mdl_indvar()} mdl_depvar= {mdl_depvar()} mdl good? {mdl() != None}")
                            res = smf.ols(f"{yv} ~ {xv}" ,data= dfg).fit() #normal operations fit a new model put try/catch here
                            MTYPE = 'OLS'
                        except:
                            res = None
                   
                sq = 0.05
                deltax = dfg[xv].max() - dfg[xv].min() 
                xlo = dfg[xv].min() - sq*deltax
                xup = dfg[xv].max() + sq*deltax
                gridcount = 25   
                xvar = np.arange(xlo, xup, deltax/gridcount)                       
                newdat = pd.DataFrame({xv : xvar})
                if res != None:
                    GOTREND = True
                    #yvar = res.predict(mdl_stringM(),data = newdat)
                    try:
                        if (MTYPE == 'LOGIT'):
                            #ynew = res.predict(exog=newdat, transform = True)
                            res_predictions = res.get_prediction(exog=newdat,transform = True)
                            res_frame = res_predictions.summary_frame(alpha = input.siglev())
                        elif(MTYPE == 'OLS'):
                            res_predictions = res.get_prediction(exog=newdat,transform = True)
                            res_frame = res_predictions.summary_frame(alpha = input.siglev())                         
                    except:
                        pushlog("Predictions failed!")
                        pushlog(" Prediction data: ")
                        pushlog(f"{newdat.head()}")
                        GOTREND = False
                    if (GOTREND):
                        if MTYPE == 'LOGIT':
                            ynew   =  res_frame['mean'].values.reshape(xvar.shape)
                            Ci_lb1 =  res_frame['mean_ci_lower'].values.reshape(xvar.shape)
                            Ci_ub1 =  res_frame['mean_ci_upper'].values.reshape(xvar.shape)
                        elif MTYPE == 'OLS':
                            ynew = res_frame['mean'].values.reshape(xvar.shape)    
                            Ci_lb1 =  res_frame['mean_ci_lower'].values.reshape(xvar.shape)
                            Ci_ub1 =  res_frame['mean_ci_upper'].values.reshape(xvar.shape)
                            Pi_lb1 =  res_frame['obs_ci_lower'].values.reshape(xvar.shape)
                            Pi_ub1 =  res_frame['obs_ci_upper'].values.reshape(xvar.shape)
                        #now plot the trendline
                        ax = sb.lineplot(x=xvar, y= ynew, color = 'red') 
                        if ('CI' in input.scttropts()):
                            ax = sb.lineplot(x = xvar, y= Ci_lb1, color = 'green')
                            ax = sb.lineplot(x = xvar, y= Ci_ub1, color = 'green')
                        if ('PI' in input.scttropts()) & (MTYPE == 'OLS' ):
                            ax = sb.lineplot(x = xvar, y= Pi_lb1, color = 'goldenrod')
                            ax = sb.lineplot(x = xvar, y= Pi_ub1, color = 'goldenrod')   
                            
            # plot extra series if needed                    
            if (input.y1var() != '-'):
                if (input.y1mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.y1var(), c = input.y1col(),  s = input.sl1())
                    pushlog(f" extra plot #1 variable= {input.y1var()} color = {input.y1col()} NOTE: missing values are dropped.")
                    #ax.add(sb.Dot())    
                elif (input.y1mark() == 'line'):
                    pushlog(f" extra plot #1  variable= {input.y1var()} color = {input.y1col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.y1var(), color = input.y1col())
                    plt.legend(title = input.y1var(), loc = 1, fontsize = 12)
            if (input.y2var() != '-'):
                if (input.y2mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.y2var(), c = input.y2col(),  s = input.sl1())
                    pushlog(f" extra plot #2 variable= {input.y2var()} color = {input.y2col()} NOTE: missing values are dropped.")  
                elif (input.y2mark() == 'line'):
                    pushlog(f" extra plot #2 variable= {input.y2var()} color = {input.y2col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.y2var(), color = input.y2col())
                    plt.legend(title = input.y2var(), loc = 1, fontsize = 12)
            if (input.y3var() != '-'):
                if (input.y3mark() == 'dot'):
                    ax=sb.scatterplot(data = dfg,x = xv,y = input.y3var(), c = input.y3col(),  s = input.sl1())
                    pushlog(f" extra plot #2 variable= {input.y23ar()} color = {input.y3col()} NOTE: missing values are dropped.")  
                elif (input.y3mark() == 'line'):
                    pushlog(f" extra plot #2 variable= {input.y3var()} color = {input.y3col()} NOTE: missing values are dropped.")
                    ax = sb.lineplot(data= dfg, x=xv, y=input.y3var(), color = input.y3col())
                    plt.legend(title = input.y3var(), loc = 1, fontsize = 12)
                    
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
        #create the row subset for graphing    
        for item in list(subdict().keys()):
            df = df[df[item].astype('str').isin(list(subdict()[item]))]
        yield df.to_csv(index = False)

    #warnings for plotting panel
    # @render.text
    # @reactive.event(input.updateB)
    # def plt_mess():
    #     return plt_msgstr()  
        
##########################################################################
####  Linear Models panel
##########################################################################
                        
    @reactive.effect
    @reactive.event(input.depvar)    
    def do_depvar():
        df = plt_data()
        #print(".....resetting model in do_depvar")
        #mdl.set(None)
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
        #print(f"At runModel Top.::::::  input.indvar = {input.indvar()}")
        size0 = len(df)
        if (input.depvar() == '-'): 
            return
        #apply the current subset items are column names items are the dictionary keys, 
        #   the dictionary entry is the list of active row values for that column.
        for item in list(subdict().keys()):
            df = df[df[item].astype('str').isin(list(subdict()[item]))]
        size1 = len(df)
        #create factors for numerical variables as set by user by changing them into strings
        if len(input.tofactor()) >0 :
            pushlog(f"runModel: Creating factors for: {', '.join(input.tofactor())}")
        for item in input.tofactor():
            df[item] = df[item].astype('str')
            
        #manually remove rows containing NaNs in the dependent or independent variables columns
        df.dropna(subset = [input.depvar()] + list(input.indvar()),inplace = True)   
        size2 = len(df)  
            
        #check to see if the dependent variable is binary (use logit) has several outcomes (use ols) or just one (quit)
        pushlog(f"runModel: {size0-size1} rows deleted by filter, {size1-size2} rows deleted due to missing data.")
        
        #minimal sanity check: a) dependent variable can't be constant b) if LOGIT has been chosen, dependent variable must be binary (0 or 1)
        outcomes = list(df[input.depvar()].unique())
        no_outcomes = len(outcomes)
        if (no_outcomes <=1):
            pushlog("The dependent variable is a constant.  I'm confused.  Please check and try again.")
            return
        STOP = False
        if (input.mtype() == 'LOGIT') & (set([0,1]) == set(outcomes)):
            try:
                #res = smf.logit(formula = input.stringM(), data=df).fit()
                res = smf.glm(formula = input.stringM(), data = df, family=sm.families.Binomial()).fit()
            except: 
                STOP = True
            mdl_type.set('LOGIT')
        elif (input.mtype() == 'OLS'):
            try:
                res = smf.ols(formula=input.stringM(), data=df).fit()
            except: 
                STOP = True            
            mdl_type.set('OLS')
        else:
            STOP = True
            pushlog("No model chosen, choose OLS or LOGIT")
        if STOP:
            mdl_indvar.set([])
            mdl.set(None)
            ui.update_radio_buttons("datachoose",choices = ['Input Data'],selected = 'Input Data')
            pushlog(f"{mdl_type()} estimation failed. Model = {input.stringM()} ")
            return
        # regression succeeded          
        mdl.set(res) 
        mdl_d = pd.concat([res.model.data.orig_exog,res.model.data.orig_endog],axis = 1)
        
        if (mdl_type() == 'LOGIT') :
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
        mdl_d = pd.concat([mdl_d,df[addoncols]],axis = 1)
        mdl_data.set(mdl_d)
        mdl_indvar.set(input.indvar())
        mdl_depvar.set(input.depvar())
        mdl_stringM.set(input.stringM())
        #now setup the plotting variables
        ui.update_select("lmsp",choices = ['-'] + list(res.params.index),selected = None)
        if (mdl_type() == 'LOGIT'):
            ui.update_radio_buttons("regplotR",choices = ['ROC', 'Partial Regression','Fit'], selected = 'ROC')
        else:
            ui.update_radio_buttons("regplotR",choices = ['Leverage','Partial Regression','Influence','Fit'],selected = 'Leverage')
        ui.update_radio_buttons("datachoose",choices = ['Input Data', 'Model Data'],selected = 'Input Data')
        pushlog(f"Estimation successful. {mdl_type()} ...Model: {mdl_stringM()}")
        pushlog(str( mdl().summary()))
        return
    
    @render.text
    @reactive.event(input.modelGo)
    def modelSummary():
        if (mdl() == None) : return f"Model estimation failed.  Check log for details. model: {input.stringM()}"
        return "Model: " + input.stringM() + "\n\n" + str( mdl().summary())
        
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
            ax.set_title(f"ROC  Model: {input.stringM()}, AUC={round(roc_auc,3)}")
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

    #warnings for linearmodels plot
    # @render.text
    # @reactive.event(input.updateM)
    # def lm_mess():
    #         return lm_msgstr()
        

#app = App(app_ui, server,debug=True)
app = App(app_ui, server)

