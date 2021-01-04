{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "## classification algo\n",
    "\n",
    "## THINGS TRIED:\n",
    "# 1. 3-class classification with SVC & RF\n",
    "\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import pickle\n",
    "from scipy import stats\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier\n",
    "from sklearn.metrics import auc,accuracy_score\n",
    "from sklearn.metrics import precision_recall_fscore_support,confusion_matrix\n",
    "\n",
    "from datetime import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "######################  functions #########################\n",
    "\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import itertools\n",
    "\n",
    "def plot_confusion_matrix(cm, classes,\n",
    "                          normalize=False,\n",
    "                          title='Confusion matrix',\n",
    "                          cmap=plt.cm.Blues):\n",
    "    \"\"\"\n",
    "    This function prints and plots the confusion matrix.\n",
    "    Normalization can be applied by setting `normalize=True`.\n",
    "    \"\"\"\n",
    "    if normalize:\n",
    "        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]\n",
    "        print(\"Normalized confusion matrix\")\n",
    "    else:\n",
    "        print('Confusion matrix, without normalization')\n",
    "\n",
    "    print(cm)\n",
    "\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes, rotation=45)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "\n",
    "    fmt = '.2f' if normalize else 'd'\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, format(cm[i, j], fmt),\n",
    "                 horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('Predicted label')\n",
    "\n",
    "\n",
    "#################################################################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#################################################################################################################\n",
    "\n",
    "### using individual epoch data ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "file_location='C:\\\\Users\\\\Desktop\\\\data\\\\HEALTHCARE\\\\EEG\\\\data\\\\Pickle_Files_6nov2018'\n",
    "os.chdir('C:\\\\Users\\\\Desktop\\\\data\\\\HEALTHCARE\\\\EEG\\\\data\\\\Pickle_Files_6nov2018') \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove xls for 2nd night of subject 13, as there was data loss \n",
    "# read in all xls, & then the sheets\n",
    "\n",
    "dfull = pickle.load(open('dfull_6nov.pkl', 'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pat_id</th>\n",
       "      <th>day</th>\n",
       "      <th>segment</th>\n",
       "      <th>epoch</th>\n",
       "      <th>delta</th>\n",
       "      <th>theta</th>\n",
       "      <th>alpha</th>\n",
       "      <th>beta</th>\n",
       "      <th>gamma</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>400</td>\n",
       "      <td>1</td>\n",
       "      <td>Seg1</td>\n",
       "      <td>1</td>\n",
       "      <td>0.703302</td>\n",
       "      <td>0.146348</td>\n",
       "      <td>0.099262</td>\n",
       "      <td>0.037832</td>\n",
       "      <td>0.018150</td>\n",
       "      <td>wake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>400</td>\n",
       "      <td>1</td>\n",
       "      <td>Seg1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.580056</td>\n",
       "      <td>0.166110</td>\n",
       "      <td>0.089898</td>\n",
       "      <td>0.058965</td>\n",
       "      <td>0.049103</td>\n",
       "      <td>wake</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  pat_id day segment epoch     delta     theta     alpha      beta     gamma  \\\n",
       "0    400   1    Seg1     1  0.703302  0.146348  0.099262  0.037832  0.018150   \n",
       "1    400   1    Seg1     2  0.580056  0.166110  0.089898  0.058965  0.049103   \n",
       "\n",
       "  class  \n",
       "0  wake  \n",
       "1  wake  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfull.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>delta</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>class</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>2804</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>17799</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>72391</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               delta\n",
       "class               \n",
       "sleep_stage_1   2804\n",
       "sleep_stage_2  17799\n",
       "wake           72391"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfull.groupby('class').agg({'delta': 'count'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>delta</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pat_id</th>\n",
       "      <th>class</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">400</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>117</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>623</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3882</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">401</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>201</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>1222</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3680</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">402</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>278</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>947</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3778</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">403</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>106</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>885</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3965</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">404</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>303</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>1134</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3307</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">405</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>158</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>833</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>4038</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">406</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>146</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>824</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>4076</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">407</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>173</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>795</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3862</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">408</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>107</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>591</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3709</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">409</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>1073</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>2818</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">410</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>182</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>1278</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3679</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">411</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>898</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3943</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">412</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>169</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3728</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">413</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>57</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>497</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>1941</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">414</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>56</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>790</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3942</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">415</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>792</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">416</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>97</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>907</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3619</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">417</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>65</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>1015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3551</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">418</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>180</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>678</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3999</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">419</th>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>190</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>1267</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>3134</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      delta\n",
       "pat_id class               \n",
       "400    sleep_stage_1    117\n",
       "       sleep_stage_2    623\n",
       "       wake            3882\n",
       "401    sleep_stage_1    201\n",
       "       sleep_stage_2   1222\n",
       "       wake            3680\n",
       "402    sleep_stage_1    278\n",
       "       sleep_stage_2    947\n",
       "       wake            3778\n",
       "403    sleep_stage_1    106\n",
       "       sleep_stage_2    885\n",
       "       wake            3965\n",
       "404    sleep_stage_1    303\n",
       "       sleep_stage_2   1134\n",
       "       wake            3307\n",
       "405    sleep_stage_1    158\n",
       "       sleep_stage_2    833\n",
       "       wake            4038\n",
       "406    sleep_stage_1    146\n",
       "       sleep_stage_2    824\n",
       "       wake            4076\n",
       "407    sleep_stage_1    173\n",
       "       sleep_stage_2    795\n",
       "       wake            3862\n",
       "408    sleep_stage_1    107\n",
       "       sleep_stage_2    591\n",
       "       wake            3709\n",
       "409    sleep_stage_1    100\n",
       "       sleep_stage_2   1073\n",
       "       wake            2818\n",
       "410    sleep_stage_1    182\n",
       "       sleep_stage_2   1278\n",
       "       wake            3679\n",
       "411    sleep_stage_1     31\n",
       "       sleep_stage_2    898\n",
       "       wake            3943\n",
       "412    sleep_stage_1    169\n",
       "       sleep_stage_2    750\n",
       "       wake            3728\n",
       "413    sleep_stage_1     57\n",
       "       sleep_stage_2    497\n",
       "       wake            1941\n",
       "414    sleep_stage_1     56\n",
       "       sleep_stage_2    790\n",
       "       wake            3942\n",
       "415    sleep_stage_1     88\n",
       "       sleep_stage_2    792\n",
       "       wake            3740\n",
       "416    sleep_stage_1     97\n",
       "       sleep_stage_2    907\n",
       "       wake            3619\n",
       "417    sleep_stage_1     65\n",
       "       sleep_stage_2   1015\n",
       "       wake            3551\n",
       "418    sleep_stage_1    180\n",
       "       sleep_stage_2    678\n",
       "       wake            3999\n",
       "419    sleep_stage_1    190\n",
       "       sleep_stage_2   1267\n",
       "       wake            3134"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfull.groupby(['pat_id','class']).agg({'delta': 'count'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "dfull['class2'] = dfull['class']\n",
    "dfull['class2'] = [0 if x == 'wake' else 1 if x == 'sleep_stage_1' else 2 for x in dfull['class2']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pat_id      object\n",
       "day         object\n",
       "segment     object\n",
       "epoch       object\n",
       "delta      float64\n",
       "theta      float64\n",
       "alpha      float64\n",
       "beta       float64\n",
       "gamma      float64\n",
       "class       object\n",
       "class2       int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfull.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "## convert selected columns to categories \n",
    "dfull[['class']] = dfull[['class']].astype('str') \n",
    "dfull[['class']] = dfull[['class']].astype('category')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8400"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### create a smaller subset of data for testing algo\n",
    "dw = dfull[(dfull['class'] == 'wake')]\n",
    "d1 = dfull[(dfull['class'] == 'sleep_stage_1')]\n",
    "d2 = dfull[(dfull['class'] == 'sleep_stage_2')]\n",
    "\n",
    "# https://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe\n",
    "# Randomly sample n elements from your dataframe\n",
    "d1_elements = d1.sample(n=2800)\n",
    "d2_elements = d2.sample(n=2800)\n",
    "dw_elements = dw.sample(n=2800)\n",
    "\n",
    "dn = pd.DataFrame()\n",
    "dn = pd.concat([dw_elements, d1_elements, d2_elements],ignore_index=True)\n",
    "dn.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6720, 5)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## create train-test data\n",
    "X = dn[['delta','theta','alpha','beta','gamma']]  # dfull[['delta','theta','alpha','beta','gamma']]\n",
    "y = dn[['class']]  # dfull[['class']] \n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)\n",
    "\n",
    "\n",
    "## normalize the data\n",
    "scaler = MinMaxScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "X_train_scaled.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#### SVC MODEL ####"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'2018-11-07 10:03:19'"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\n",
       "  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',\n",
       "  max_iter=-1, probability=False, random_state=None, shrinking=True,\n",
       "  tol=0.001, verbose=False)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## fit model - vary ML algo & hyperparameters\n",
    "\n",
    "model_svc = SVC()  # kernel='linear'\n",
    "model_svc.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'2018-11-07 10:03:21'"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# predict on test data - check metrics\n",
    "y_pred = model_svc.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion matrix, without normalization\n",
      "[[296 105 145]\n",
      " [ 70 354 134]\n",
      " [ 54  15 507]]\n",
      "Normalized confusion matrix\n",
      "[[0.54 0.19 0.27]\n",
      " [0.13 0.63 0.24]\n",
      " [0.09 0.03 0.88]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "####    Compute confusion matrix     ####\n",
    "\n",
    "class_names = ['wake','sleep_stage_1','sleep_stage_2']  # wake, SS1, SS2  ; # '0','1','2'\n",
    "\n",
    "cnf_matrix = confusion_matrix(y_test, y_pred)\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names,\n",
    "                      title='Confusion matrix, without normalization')\n",
    "\n",
    "# Plot normalized confusion matrix : normalisation shows nan for class'0' no signal has class=0 as true label\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,\n",
    "                      title='Normalized confusion matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "               precision    recall  f1-score   support\n",
      "\n",
      "         wake       0.70      0.54      0.61       546\n",
      "sleep_stage_1       0.75      0.63      0.69       558\n",
      "sleep_stage_2       0.65      0.88      0.74       576\n",
      "\n",
      "  avg / total       0.70      0.69      0.68      1680\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>class_act</th>\n",
       "      <th>class_pred</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2876</th>\n",
       "      <td>sleep_stage_1</td>\n",
       "      <td>sleep_stage_1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3281</th>\n",
       "      <td>sleep_stage_1</td>\n",
       "      <td>sleep_stage_2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          class_act     class_pred\n",
       "2876  sleep_stage_1  sleep_stage_1\n",
       "3281  sleep_stage_1  sleep_stage_2"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts = pd.DataFrame()\n",
    "Ts['class_act'] = y_test['class']\n",
    "Ts['class_pred'] = y_pred\n",
    "Ts.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_act\n",
       "sleep_stage_1    546\n",
       "sleep_stage_2    558\n",
       "wake             576\n",
       "dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts.groupby('class_act').size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_pred\n",
       "sleep_stage_1    420\n",
       "sleep_stage_2    474\n",
       "wake             786\n",
       "dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts.groupby('class_pred').size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "### comment: although the row counts are correct, the labels are not correct in the confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#### RF model ####"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'2018-11-07 12:04:34'"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py:3: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  This is separate from the ipykernel package so we can avoid doing imports until\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n",
       "            max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
       "            min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "            min_samples_leaf=1, min_samples_split=2,\n",
       "            min_weight_fraction_leaf=0.0, n_estimators=501, n_jobs=1,\n",
       "            oob_score=False, random_state=None, verbose=0,\n",
       "            warm_start=False)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_rf = RandomForestClassifier(n_estimators = 501) ## max_depth=5, random_state=0,verbose =0)  \n",
    "# max_features=2, min_samples_split=4, n_estimators=50, min_samples_leaf=2\n",
    "model_rf.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'2018-11-07 12:04:42'"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# predict on test data - check metrics\n",
    "y_pred = model_rf.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "               precision    recall  f1-score   support\n",
      "\n",
      "         wake       0.69      0.63      0.66       546\n",
      "sleep_stage_1       0.75      0.78      0.76       558\n",
      "sleep_stage_2       0.78      0.82      0.80       576\n",
      "\n",
      "  avg / total       0.74      0.74      0.74      1680\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion matrix, without normalization\n",
      "[[346 117  83]\n",
      " [ 76 435  47]\n",
      " [ 78  28 470]]\n",
      "Normalized confusion matrix\n",
      "[[0.63 0.21 0.15]\n",
      " [0.14 0.78 0.08]\n",
      " [0.14 0.05 0.82]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "####    Compute confusion matrix     ####\n",
    "\n",
    "class_names = ['wake','sleep_stage_1','sleep_stage_2']  # wake, SS1, SS2  ; # '0','1','2'\n",
    "\n",
    "cnf_matrix = confusion_matrix(y_test, y_pred)\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names,\n",
    "                      title='Confusion matrix, without normalization')\n",
    "\n",
    "# Plot normalized confusion matrix : normalisation shows nan for class'0' no signal has class=0 as true label\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,\n",
    "                      title='Normalized confusion matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>class_act</th>\n",
       "      <th>class_pred</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2876</th>\n",
       "      <td>sleep_stage_1</td>\n",
       "      <td>sleep_stage_1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3281</th>\n",
       "      <td>sleep_stage_1</td>\n",
       "      <td>sleep_stage_2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          class_act     class_pred\n",
       "2876  sleep_stage_1  sleep_stage_1\n",
       "3281  sleep_stage_1  sleep_stage_2"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts = pd.DataFrame()\n",
    "Ts['class_act'] = y_test['class']\n",
    "Ts['class_pred'] = y_pred\n",
    "Ts.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_act\n",
       "sleep_stage_1    546\n",
       "sleep_stage_2    558\n",
       "wake             576\n",
       "dtype: int64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts.groupby('class_act').size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_pred\n",
       "sleep_stage_1    500\n",
       "sleep_stage_2    580\n",
       "wake             600\n",
       "dtype: int64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts.groupby('class_pred').size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion matrix, without normalization\n",
      "[[346 117  83]\n",
      " [ 76 435  47]\n",
      " [ 78  28 470]]\n",
      "Normalized confusion matrix\n",
      "[[0.63 0.21 0.15]\n",
      " [0.14 0.78 0.08]\n",
      " [0.14 0.05 0.82]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## in class names, use the same order as in output of groupby \n",
    "class_names = ['sleep_stage_1','sleep_stage_2','wake']  # wake, SS1, SS2  ; # '0','1','2'\n",
    "\n",
    "cnf_matrix = confusion_matrix(y_test, y_pred)\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names,\n",
    "                      title='Confusion matrix, without normalization')\n",
    "\n",
    "# Plot normalized confusion matrix : normalisation shows nan for class'0' no signal has class=0 as true label\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,\n",
    "                      title='Normalized confusion matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0       0.69      0.63      0.66       546\n",
      "          1       0.75      0.78      0.76       558\n",
      "          2       0.78      0.82      0.80       576\n",
      "\n",
      "avg / total       0.74      0.74      0.74      1680\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, y_pred, target_names=class_names))\n",
    "# 0 corrs to SS1,  1 : SS2  &  3: wake"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## change the classification from wake, SS1 etc to 0,1, 2 & check precision,recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "Ts['class_act'] = ['0' if x == 'wake' else '1' if x == 'sleep_stage_1' else '2' for x in Ts['class_act']]\n",
    "Ts['class_pred'] = ['0' if x == 'wake' else '1' if x == 'sleep_stage_1' else '2' for x in Ts['class_pred']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_act\n",
       "0    576\n",
       "1    546\n",
       "2    558\n",
       "dtype: int64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts.groupby('class_act').size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_pred\n",
       "0    600\n",
       "1    500\n",
       "2    580\n",
       "dtype: int64"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts.groupby('class_pred').size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0       0.78      0.82      0.80       576\n",
      "          1       0.69      0.63      0.66       546\n",
      "          2       0.75      0.78      0.76       558\n",
      "\n",
      "avg / total       0.74      0.74      0.74      1680\n",
      "\n"
     ]
    }
   ],
   "source": [
    "## from RF model\n",
    "class_names = ['0','1','2']\n",
    "ytest = Ts['class_act']\n",
    "ypred = Ts['class_pred']\n",
    "print(classification_report(ytest, ypred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion matrix, without normalization\n",
      "[[470  78  28]\n",
      " [ 83 346 117]\n",
      " [ 47  76 435]]\n",
      "Normalized confusion matrix\n",
      "[[0.82 0.14 0.05]\n",
      " [0.15 0.63 0.21]\n",
      " [0.08 0.14 0.78]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## in class names, use the same order as in output of groupby \n",
    "class_names = ['0','1','2']  # wake, SS1, SS2  ; # '0','1','2'\n",
    "\n",
    "cnf_matrix = confusion_matrix(ytest, ypred)\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names,\n",
    "                      title='Confusion matrix, without normalization')\n",
    "\n",
    "# Plot normalized confusion matrix : normalisation shows nan for class'0' no signal has class=0 as true label\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,\n",
    "                      title='Normalized confusion matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>delta</th>\n",
       "      <th>theta</th>\n",
       "      <th>alpha</th>\n",
       "      <th>beta</th>\n",
       "      <th>gamma</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2586</th>\n",
       "      <td>0.550596</td>\n",
       "      <td>0.176619</td>\n",
       "      <td>0.131146</td>\n",
       "      <td>0.078002</td>\n",
       "      <td>0.032223</td>\n",
       "      <td>wake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5467</th>\n",
       "      <td>0.461529</td>\n",
       "      <td>0.167010</td>\n",
       "      <td>0.148383</td>\n",
       "      <td>0.116500</td>\n",
       "      <td>0.065478</td>\n",
       "      <td>sleep_stage_1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4576</th>\n",
       "      <td>0.678690</td>\n",
       "      <td>0.107442</td>\n",
       "      <td>0.066313</td>\n",
       "      <td>0.056259</td>\n",
       "      <td>0.045225</td>\n",
       "      <td>sleep_stage_1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5766</th>\n",
       "      <td>0.423911</td>\n",
       "      <td>0.202512</td>\n",
       "      <td>0.138790</td>\n",
       "      <td>0.131020</td>\n",
       "      <td>0.070190</td>\n",
       "      <td>sleep_stage_2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6751</th>\n",
       "      <td>0.610648</td>\n",
       "      <td>0.234870</td>\n",
       "      <td>0.098794</td>\n",
       "      <td>0.040527</td>\n",
       "      <td>0.016132</td>\n",
       "      <td>sleep_stage_2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         delta     theta     alpha      beta     gamma          class\n",
       "2586  0.550596  0.176619  0.131146  0.078002  0.032223           wake\n",
       "5467  0.461529  0.167010  0.148383  0.116500  0.065478  sleep_stage_1\n",
       "4576  0.678690  0.107442  0.066313  0.056259  0.045225  sleep_stage_1\n",
       "5766  0.423911  0.202512  0.138790  0.131020  0.070190  sleep_stage_2\n",
       "6751  0.610648  0.234870  0.098794  0.040527  0.016132  sleep_stage_2"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## save the training & test sets\n",
    "\n",
    "Train = pd.concat([X_train,y_train],axis=1)\n",
    "Train.to_excel(\"C:\\\\Users\\\\Desktop\\\\data\\\\HEALTHCARE\\\\EEG\\\\data\\\\training_data_3stage.xlsx\")\n",
    "\n",
    "Train.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>delta</th>\n",
       "      <th>theta</th>\n",
       "      <th>alpha</th>\n",
       "      <th>beta</th>\n",
       "      <th>gamma</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2876</th>\n",
       "      <td>0.493560</td>\n",
       "      <td>0.102239</td>\n",
       "      <td>0.102969</td>\n",
       "      <td>0.096628</td>\n",
       "      <td>0.115924</td>\n",
       "      <td>sleep_stage_1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3281</th>\n",
       "      <td>0.532330</td>\n",
       "      <td>0.188965</td>\n",
       "      <td>0.132549</td>\n",
       "      <td>0.078836</td>\n",
       "      <td>0.041364</td>\n",
       "      <td>sleep_stage_1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2631</th>\n",
       "      <td>0.583887</td>\n",
       "      <td>0.120817</td>\n",
       "      <td>0.075969</td>\n",
       "      <td>0.065158</td>\n",
       "      <td>0.070059</td>\n",
       "      <td>wake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1837</th>\n",
       "      <td>0.397956</td>\n",
       "      <td>0.303622</td>\n",
       "      <td>0.151444</td>\n",
       "      <td>0.077125</td>\n",
       "      <td>0.047845</td>\n",
       "      <td>wake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3535</th>\n",
       "      <td>0.642075</td>\n",
       "      <td>0.093682</td>\n",
       "      <td>0.088097</td>\n",
       "      <td>0.093259</td>\n",
       "      <td>0.051846</td>\n",
       "      <td>sleep_stage_1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         delta     theta     alpha      beta     gamma          class\n",
       "2876  0.493560  0.102239  0.102969  0.096628  0.115924  sleep_stage_1\n",
       "3281  0.532330  0.188965  0.132549  0.078836  0.041364  sleep_stage_1\n",
       "2631  0.583887  0.120817  0.075969  0.065158  0.070059           wake\n",
       "1837  0.397956  0.303622  0.151444  0.077125  0.047845           wake\n",
       "3535  0.642075  0.093682  0.088097  0.093259  0.051846  sleep_stage_1"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Test = pd.concat([X_test,y_test],axis=1)\n",
    "Test.to_excel(\"C:\\\\Users\\\\Desktop\\\\data\\\\HEALTHCARE\\\\EEG\\\\data\\\\test_data_3stage.xlsx\")\n",
    "\n",
    "Test.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>delta</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>class</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>2254</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>2242</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>2224</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               delta\n",
       "class               \n",
       "sleep_stage_1   2254\n",
       "sleep_stage_2   2242\n",
       "wake            2224"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Train.groupby('class').agg({'delta': 'count'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>delta</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>class</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>sleep_stage_1</th>\n",
       "      <td>546</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sleep_stage_2</th>\n",
       "      <td>558</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>wake</th>\n",
       "      <td>576</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               delta\n",
       "class               \n",
       "sleep_stage_1    546\n",
       "sleep_stage_2    558\n",
       "wake             576"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Test.groupby('class').agg({'delta': 'count'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "########################################################################################\n",
    "\n",
    "## using the class labels as numeric"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_act     object\n",
       "class_pred    object\n",
       "dtype: object"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_act     int32\n",
       "class_pred    int32\n",
       "dtype: object"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts[['class_act','class_pred']] = Ts[['class_act','class_pred']].astype(int)\n",
    "Ts.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "object of type 'int' has no len()",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-49-fbc14b78995f>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mytest\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mTs\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'class_act'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mypred\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mTs\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'class_pred'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mclassification_report\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mytest\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mypred\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtarget_names\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mclass_names\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\sklearn\\metrics\\classification.py\u001b[0m in \u001b[0;36mclassification_report\u001b[1;34m(y_true, y_pred, labels, target_names, sample_weight, digits)\u001b[0m\n\u001b[0;32m   1433\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mtarget_names\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1434\u001b[0m         \u001b[0mtarget_names\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;34mu'%s'\u001b[0m \u001b[1;33m%\u001b[0m \u001b[0ml\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0ml\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mlabels\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1435\u001b[1;33m     \u001b[0mname_width\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmax\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcn\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mcn\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtarget_names\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1436\u001b[0m     \u001b[0mwidth\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmax\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mname_width\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlast_line_heading\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdigits\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1437\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\sklearn\\metrics\\classification.py\u001b[0m in \u001b[0;36m<genexpr>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m   1433\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mtarget_names\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1434\u001b[0m         \u001b[0mtarget_names\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;34mu'%s'\u001b[0m \u001b[1;33m%\u001b[0m \u001b[0ml\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0ml\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mlabels\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1435\u001b[1;33m     \u001b[0mname_width\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmax\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcn\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mcn\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtarget_names\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1436\u001b[0m     \u001b[0mwidth\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmax\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mname_width\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlast_line_heading\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdigits\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1437\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: object of type 'int' has no len()"
     ]
    }
   ],
   "source": [
    "## from RF model\n",
    "class_names = [0,1,2]\n",
    "ytest = Ts['class_act']\n",
    "ypred = Ts['class_pred']\n",
    "print(classification_report(ytest, ypred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion matrix, without normalization\n",
      "[[470  78  28]\n",
      " [ 83 346 117]\n",
      " [ 47  76 435]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cnf_matrix = confusion_matrix(ytest, ypred)\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names,\n",
    "                      title='Confusion matrix, without normalization')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "Ts['class_act'] = [0 if x == '0' else 1 if x == '1' else 2 for x in Ts['class_act']]\n",
    "Ts['class_pred'] = [0 if x == '0' else 1 if x == '1' else 2 for x in Ts['class_pred']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "class_act     int64\n",
       "class_pred    int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ts.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0       1.00      1.00      1.00      1680\n",
      "\n",
      "avg / total       1.00      1.00      1.00      1680\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\sklearn\\metrics\\classification.py:1428: UserWarning: labels size, 1, does not match size of target_names, 3\n",
      "  .format(len(labels), len(target_names))\n"
     ]
    }
   ],
   "source": [
    "## from RF model\n",
    "class_names = ['0','1','2']\n",
    "ytest = Ts['class_act']\n",
    "ypred = Ts['class_pred']\n",
    "print(classification_report(ytest, ypred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
