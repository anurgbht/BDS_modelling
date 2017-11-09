import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import pandas as pd
import numpy as np
import implement_functions as mf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
import pickle

def include_tree(row):
    x3 = row[2]
    x10 = row[9]
    x16 = row[15]
    x14 = row[13]
    x17 = row[16]
    x3 = row[2]

    if x3 <= 8.345:
        if x3 <= 5.25:
            ## A
            tt = -4.627443
        elif x3 > 5.25:
            if x14 <= -1.445:
                if x17 <= 0.045:
                    ## B
                    tt = 2.141884
                elif x17 > 0.045:
                    ## C
                    tt = 0.247894
            elif x14 > -1.445:
                ## D
                tt = -1.716035
    elif x3 > 8.345:
        if x16 <= 9.45:
            if x14 <= -0.405:
                ## E
                tt = 1.991644
            elif x14 > -0.405:
                ## F
                tt = 0.043099
        elif x16 > 9.45:
            ## G
            tt = 4.45875
        
    return tt

def subs_WOE(row):
    temp = woe_table.loc[woe_table.iloc[:,0] == row.iloc[-1],1]
    return temp

def my_pred_fun(beta_0,beta,temp_x_trigger):
    temp = pd.DataFrame()
    for i in range(temp_x_trigger.shape[0]):
        exp_temp = np.exp(beta_0 + np.inner(beta,temp_x_trigger.iloc[i,:].tolist()))
        temp1 = 1/(1+exp_temp)
        temp = temp.append([temp1])
    temp = list(temp.iloc[:,0])
    b=list(np.zeros(60))
    b.extend(temp)
    b = pd.DataFrame(b)
    return b


class App:
    def __init__(self, master):

        os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')
        with open("logistic_pickle_backup.pickle",'rb') as f:
            clf_logit_trigger = pickle.load(f)
            f.close()

        beta_0 = clf_logit_trigger.intercept_
        global beta
        beta = clf_logit_trigger.coef_[0].tolist()
        beta_0_back = clf_logit_trigger.intercept_
        global beta_back
        beta_back = clf_logit_trigger.coef_[0].tolist()
        
        print('Initial Values')
        print(beta_0,beta)

        file_name = "D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/breakout files/d5122018.csv"
        title_name = file_name.split('/')[-1]
        file = pd.read_csv(file_name)
        file = file.iloc[2500:3000,:].reset_index()
##        TC_layer = int(title_name.split('_')[4]) + 1
        TC_layer = 15
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)
        tt3 = 'TC' + str(TC_layer + 40)

        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]
        L3 = file.loc[:,tt3]
        ML = file.loc[:,'M.level']
        CS = file.loc[:,'C.speed']
        CP = file.loc[:,'C.percent']
        MW = file.loc[:,'M.width']

        TC_layer_left = mf.find_left(MW[0],TC_layer)

        tt1 = 'TC' + str(TC_layer_left)
        tt2 = 'TC' + str(TC_layer_left + 20)

        LL1 = file.loc[:,tt1]
        LL2 = file.loc[:,tt2]

        TC_layer_right = mf.find_right(MW[0],TC_layer)

        tt1 = 'TC' + str(TC_layer_right)
        tt2 = 'TC' + str(TC_layer_right + 20)

        LR1 = file.loc[:,tt1]
        LR2 = file.loc[:,tt2]

        # making the continuous x's
        temp_x_trigger1 = mf.make_cont_x(L1,L2,ML,CP,CS)
        # including the tree based information
        temp1 = temp_x_trigger1.apply(include_tree,axis=1)
        temp_x_trigger1 = pd.concat([temp_x_trigger1,temp1],axis=1)

        logit_trigger1 = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_trigger1))
        logit_trigger1 = list(logit_trigger1.iloc[:,-1])
        b1 = list(np.zeros(60))
        b1.extend(logit_trigger1)
        b1 = pd.DataFrame(b1)

        # making the continuous x's
        temp_x_trigger2 = mf.make_cont_x(L2,L3,ML,CP,CS)
        # including the tree based information
        temp2 = temp_x_trigger2.apply(include_tree,axis=1)
        temp_x_trigger2 = pd.concat([temp_x_trigger2,temp2],axis=1)

        logit_trigger2 = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_trigger2))
        logit_trigger2 = list(logit_trigger2.iloc[:,-1])
        b2 = list(np.zeros(60))
        b2.extend(logit_trigger2)
        b2 = pd.DataFrame(b2)
        
        # Create a container
        frame = tkinter.Frame(master)

        var = tkinter.StringVar(frame)
        var.set("Which feature do you want") # initial value
        choices = ["1: slope_n(L2,3)","2: slope_n(L2,5)","3: slope_n(L2,7)","4: sign_present(L1,L2,2)","5: sign_present(L1,L2,5)","6: sign_present(L1,L2,7)","7: last_n(L1,5)","8: last_n(L2,5)","9: last_n(L1,32)","10: last_n(L2,32)","11: CP","12: np.std(L1.iloc[-12:])","13: np.std(ML.iloc[-12:])","14: find_l1_peak_slope(L1,CS)","15: find_l1_peak_ratio(L1,CS)","16: find_drop(L1)","17: find_drop(L2)","18: cs_change(CS)","19: get_kinks(L2)","20: first_derivative(L2)"]
        option = tkinter.OptionMenu(frame,var,*choices)
        option.pack(side = tkinter.LEFT)

        slider = tkinter.Scale(frame, from_=-10, to=10, resolution=0.01, orient=tkinter.HORIZONTAL,length = 750)
        slider.pack(side = tkinter.LEFT)
        
        self.button_select = tkinter.Button(frame,text="Select the value ?",
                                        command=lambda:self.select(var.get(),slider.get(),beta_0,beta,temp_x_trigger1,temp_x_trigger2,choices))
        self.button_select.pack(side = tkinter.LEFT)

        self.button_reset = tkinter.Button(frame,text = "Reset",command=lambda:self.reset_func(clf_logit_trigger,beta_0,temp_x_trigger1,temp_x_trigger2)).pack(side = tkinter.LEFT)
        self.button_print_beta_all= tkinter.Button(frame,text="Print All Beta",command=self.print_beta_all).pack(side = tkinter.LEFT)
        self.button_print_beta_selected= tkinter.Button(frame,text="Print Selected Beta",command=lambda:self.print_beta_selected(choices,var.get())).pack(side = tkinter.LEFT)

##        self.textBox = tkinter.Text(frame,height=10, width = 25)
##        self.textBox.insert(tkinter.END,beta)
##        self.textBox.config(state=tkinter.DISABLED)
##        self.textBox.pack(side = tkinter.BOTTOM)

        fig = Figure()
        fig.suptitle(title_name)
        
        ax = fig.add_subplot(411)
        ax.set_ylim(0,1)
        self.line, = ax.plot(range(len(b1)),b1)
        ax.grid(b=True, which='both')

        ax2 = fig.add_subplot(412)
        ax2.set_ylim(0,1)
        self.line2, = ax2.plot(range(len(b2)),b2)
        ax2.grid(b=True, which='both')
        
        ax3 = fig.add_subplot(413)
        ax3.plot(range(len(L1)),L1,range(len(L2)),L2,range(len(L3)),L3)
        ax3.legend(["L1","L2","L3"])
        ax3.grid(b=True, which='both')

        ax4 = fig.add_subplot(414)
        ax4.plot(range(len(ML)),ML,range(len(CS)),100*CS)
        ax4.legend(["Mold Level","100x Casting Speed"])
        ax4.grid(b=True, which='both')
        
        self.canvas = FigureCanvasTkAgg(fig,master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()

    def select(self,beta_name,beta_value,beta_0,beta,temp_x_trigger1,temp_x_trigger2,choices):
        index_val = choices.index(beta_name)
        beta[index_val]= float(beta_value)
        yList = my_pred_fun(beta_0,beta,temp_x_trigger1)
        yList2 = my_pred_fun(beta_0,beta,temp_x_trigger2)
        self.line.set_ydata(yList)
        self.line2.set_ydata(yList2)
        self.canvas.draw()

    def reset_func(self,clf_logit_trigger,beta_0,temp_x_trigger1,temp_x_trigger2):
        global beta
        print('Called reset_func')
        beta = clf_logit_trigger.coef_[0].tolist()
        yList1 = my_pred_fun(beta_0,beta,temp_x_trigger1)
        yList2 = my_pred_fun(beta_0,beta,temp_x_trigger2)
        self.line.set_ydata(yList1)
        self.line2.set_ydata(yList2)
        self.canvas.draw()

    def print_beta_all(self):
        print('Called print_beta_all')
        print(beta)
        print(beta_back)

    def print_beta_selected(self,choices,beta_name):
        global beta_back
        print('Called print_beta_selected')
        index_val = choices.index(beta_name)
        print(beta_name,beta_back[index_val])

####################################################################################
####################################################################################
####################################################################################

root = tkinter.Tk()
root.title("My Visualization Tool")
app = App(root)
root.mainloop()
