# Dacon 행동분석 경진대회 참여 정리

*대회 홈페이지 : [월간 데이콘 3 행동 데이터 분석 대회](https://dacon.io/competitions/official/235583/overview/)*

- 목표 : Blizzard 스타크레프트 2 경기 행동 데이터로 승자 예측하기
- 평가 : AUC score
- 기간 : 2020.03.01 ~ 2020.04.15
- 결과 : 7위
- 코드 공유 : [Kaggle Link](www.kaggle.com/dataset/b369d2abc34ae1f98490d105bbffcd7f0fac878cfbf9a934a6ef04f7b63a99cc)

## Data Description

## Overview

- https://github.com/Blizzard/s2client-proto#downloads 의 스타크레프트2 리플레이를 csv로 저장된 데이터가 제공되었습니다.
- 데이터는 시간변화에 따른 플레이어의 행동 정보를 담고 있으며, 약 5만여개의 게임으로 이루어져있습니다.
- 데이터는 두개의 그룹으로 나눠져 있으며, 각 그룹의 용량은 다음과 같습니다. (train : 4GB,  test : 1.6GB)
- 특이점 : 전체 게임 데이터가 아닌 **일부가 무작위로 제거된 데이터**가 제공되었습니다.


```python
#Import library
import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 50)

#Load dataset
data = pd.read_pickle('../input/dacon-game/train.pkl')
data.head()
```

데이터는 총 7개의 columns을 가지고 있으며 각 type은 다음과 같습니다. 데이터의 용량을 줄이기 위해 `int`, `float`의 format을 변경해서 원본 데이터와 차이가 있지만 전체적인 type은 동일합니다.


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 67091776 entries, 0 to 67091775
    Data columns (total 7 columns):
    game_id           uint16
    winner            uint8
    time              float32
    player            uint8
    species           uint8
    event             uint8
    event_contents    object
    dtypes: float32(1), object(1), uint16(1), uint8(4)
    memory usage: 1.1+ GB
    

### Input data

제공된 데이터를 가공해서 전체 게임 로그를 한줄의 rows로 만들고, 1번 플레이어가 이길 확률값을 예측하는 모델을 구성하기로 했습니다. 

최종 제출 당시, 상관계수(`corr`)를 이용한 feature selection을 적용하여 변수 간 상관계수가 높은 변수는 제외하였습니다.

# Features

변수는 각 플레이어마다 계산한 값(`_0` & `_1`)과 두 플레이어간 차이(`_delta`)를 계산한 값을 함께 사용했습니다. 

## 1. Basic features

- `species` : 종족 변수는 각 플레이어의 고유 종족값(`t:0, p:1, z:2`)과 두 플레이어의 종족을 하나로 묶어(ex: `T vs T : 0`, `T vs Z : 2`, `Z vs T : 2`) 총 7개로 분류한 category 변수를 추가해서 사용했습니다. 
- `map` : 스타크레프트에는 맵의 종류가 한정되어 있습니다. 어떠한 맵에서 게임이 진행되었는지 정보가 제공되지않았지만 `event:camere`의 첫번째 x,y 좌표값을 kmean clustering 알고리즘으로 분류한 결과값과 [kmeans를 이용한 스타팅포인트,맵 클러스터링](https://dacon.io/competitions/official/235583/codeshare/743)
의 내용을 참고하여 7개의 map과 알수없는 데이터를 따로 분류하여 총 8개의 category 변수로 만들었습니다.
- `starting point` : 각 map에는 고유한 starting point가 있으며, `event:camere`의 첫번째 x,y 좌표값과 map 정보를 이용해서 해당 플레이어의 시작 위치를 총 16개 category 변수로 만들었습니다. 

## 2. Event Features

`event`는 총 8가지 event가 있으며, event_cotents에서 해당 event의 자세한 내용을 알 수 있습니다. 주로 이용한 event는 `camera : 0`, `right click:1`, `ability : 2`, `Selection : 3` 입니다.

## 2-1. Basic features
- 각 event의 count

## 2-2. Vision(Coordinate) features

특정 event의 event_contents에는 x,y 좌표값이 적혀있습니다. `pandas.str.extract`과 정규 표현식으로 x,y 좌표값을 추출하여 특정 연산을 수행했습니다.
    
    
    data['event_contents'].str.extract(r'[Location|At|at].+[(](?P<x>[\w.]+)[\W,]+(?P<y>[\w.]+)')

1. `euc_c / euc_t` : 연속되는 좌표 사이의 유클리드 거리를 계산한 값의 합을 구합니다(`sum`). 해당 값을 좌표 데이터의 갯수(count)나 해당 game의 마지막 time으로 나눠준 값입니다.
2. `Start_vision_var` : 시작 위치를 기준으로 좌표의 분산을 계산합니다.
3. `Other_vision_var` : 상대편 시작 위치를 기준으로 좌표의 분산을 계산합니다.
4. `Other_vision_point` : 상대편 시작 위치의 +-30에 위치한 좌표의 갯수(`count`)를 계산합니다.

좌표 관련 변수는 게임 전체를 대상으로 계산한 변수와 게임 후반(`time`기준 마지막 20%) 데이터를 대상으로 계산한 값을 사용했으며, 비율(`후반 데이터 / 전체 데이터`)을 계산한 값을 추가했습니다.


```python
train_vision = pd.read_csv('../input/dacon-game/train_vision_v6.csv')
train_vision.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_id</th>
      <th>euc_c_0</th>
      <th>euc_t_0</th>
      <th>start_var_0</th>
      <th>other_p_0</th>
      <th>other_var_0</th>
      <th>euc_c_NC_0</th>
      <th>euc_t_NC_0</th>
      <th>start_var_NC_0</th>
      <th>other_p_NC_0</th>
      <th>other_var_NC_0</th>
      <th>euc_c_RC_0</th>
      <th>euc_t_RC_0</th>
      <th>start_var_RC_0</th>
      <th>other_p_RC_0</th>
      <th>other_var_RC_0</th>
      <th>euc_c_last_0</th>
      <th>euc_t_last_0</th>
      <th>start_var_last_0</th>
      <th>other_p_last_0</th>
      <th>other_var_last_0</th>
      <th>euc_c_NC_last_0</th>
      <th>euc_t_NC_last_0</th>
      <th>start_var_NC_last_0</th>
      <th>other_p_NC_last_0</th>
      <th>...</th>
      <th>euc_c_NC_ear_delta</th>
      <th>euc_t_NC_ear_delta</th>
      <th>start_var_NC_ear_delta</th>
      <th>other_p_NC_ear_delta</th>
      <th>other_var_NC_ear_delta</th>
      <th>euc_c_RC_ear_delta</th>
      <th>euc_t_RC_ear_delta</th>
      <th>start_var_RC_ear_delta</th>
      <th>other_p_RC_ear_delta</th>
      <th>other_var_RC_ear_delta</th>
      <th>euc_c_rate_delta</th>
      <th>euc_t_rate_delta</th>
      <th>start_var_rate_delta</th>
      <th>other_p_rate_delta</th>
      <th>other_var_rate_delta</th>
      <th>euc_c_NC_rate_delta</th>
      <th>euc_t_NC_rate_delta</th>
      <th>start_var_NC_rate_delta</th>
      <th>other_p_NC_rate_delta</th>
      <th>other_var_NC_rate_delta</th>
      <th>euc_c_RC_rate_delta</th>
      <th>euc_t_RC_rate_delta</th>
      <th>start_var_RC_rate_delta</th>
      <th>other_p_RC_rate_delta</th>
      <th>other_var_RC_rate_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4.8602</td>
      <td>325.8433</td>
      <td>55.922831</td>
      <td>42.0</td>
      <td>152.547269</td>
      <td>27.8847</td>
      <td>192.6118</td>
      <td>89.280488</td>
      <td>9.0</td>
      <td>133.101789</td>
      <td>34.2957</td>
      <td>162.5757</td>
      <td>102.102850</td>
      <td>8.0</td>
      <td>123.315184</td>
      <td>10.3381</td>
      <td>16.8023</td>
      <td>35.251912</td>
      <td>0.0</td>
      <td>150.585479</td>
      <td>10.3381</td>
      <td>16.8023</td>
      <td>35.251912</td>
      <td>0.0</td>
      <td>...</td>
      <td>27.5333</td>
      <td>193.4175</td>
      <td>72.537360</td>
      <td>3.0</td>
      <td>-33.672269</td>
      <td>40.1427</td>
      <td>205.4390</td>
      <td>98.652826</td>
      <td>3.0</td>
      <td>-52.711195</td>
      <td>-7.598444</td>
      <td>-0.541572</td>
      <td>-1.436866</td>
      <td>-0.071429</td>
      <td>0.119837</td>
      <td>-1.427024</td>
      <td>-0.831888</td>
      <td>-1.105403</td>
      <td>-0.333333</td>
      <td>0.132792</td>
      <td>-1.056791</td>
      <td>-1.238142</td>
      <td>-1.489037</td>
      <td>-0.375000</td>
      <td>0.305921</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.3376</td>
      <td>606.3615</td>
      <td>54.669695</td>
      <td>22.0</td>
      <td>160.555249</td>
      <td>15.7431</td>
      <td>317.5342</td>
      <td>77.283883</td>
      <td>9.0</td>
      <td>150.624292</td>
      <td>17.5191</td>
      <td>285.5398</td>
      <td>81.669967</td>
      <td>8.0</td>
      <td>147.377738</td>
      <td>18.3490</td>
      <td>121.4960</td>
      <td>125.043940</td>
      <td>9.0</td>
      <td>98.692601</td>
      <td>18.3490</td>
      <td>121.4960</td>
      <td>125.043940</td>
      <td>9.0</td>
      <td>...</td>
      <td>-19.9700</td>
      <td>-261.4128</td>
      <td>-51.728796</td>
      <td>-10.0</td>
      <td>29.782274</td>
      <td>-20.4780</td>
      <td>-231.5440</td>
      <td>-58.138031</td>
      <td>-9.0</td>
      <td>32.648208</td>
      <td>1.818123</td>
      <td>0.302520</td>
      <td>1.552181</td>
      <td>0.500834</td>
      <td>-0.361801</td>
      <td>0.719040</td>
      <td>0.547810</td>
      <td>1.067244</td>
      <td>1.555556</td>
      <td>-0.383118</td>
      <td>0.813759</td>
      <td>0.599502</td>
      <td>1.067979</td>
      <td>1.529412</td>
      <td>-0.429209</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>12.8289</td>
      <td>786.4386</td>
      <td>62.447086</td>
      <td>42.0</td>
      <td>135.480053</td>
      <td>18.7014</td>
      <td>372.7797</td>
      <td>65.824098</td>
      <td>13.0</td>
      <td>135.263230</td>
      <td>19.8754</td>
      <td>318.5370</td>
      <td>69.052740</td>
      <td>11.0</td>
      <td>134.022412</td>
      <td>16.3575</td>
      <td>86.8397</td>
      <td>71.232834</td>
      <td>5.0</td>
      <td>130.861454</td>
      <td>16.3575</td>
      <td>86.8397</td>
      <td>71.232834</td>
      <td>5.0</td>
      <td>...</td>
      <td>-0.9062</td>
      <td>-90.5388</td>
      <td>-6.840320</td>
      <td>-8.0</td>
      <td>-2.743931</td>
      <td>-0.3151</td>
      <td>-109.1666</td>
      <td>-4.080929</td>
      <td>-8.0</td>
      <td>-4.318979</td>
      <td>0.136870</td>
      <td>0.056069</td>
      <td>-0.080839</td>
      <td>0.171131</td>
      <td>0.192990</td>
      <td>0.053477</td>
      <td>-0.043293</td>
      <td>-0.060381</td>
      <td>0.546154</td>
      <td>0.195807</td>
      <td>0.236019</td>
      <td>0.077829</td>
      <td>-0.184743</td>
      <td>0.427273</td>
      <td>0.272412</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8.1912</td>
      <td>839.3595</td>
      <td>69.797052</td>
      <td>200.0</td>
      <td>118.256354</td>
      <td>12.5412</td>
      <td>390.0782</td>
      <td>79.695175</td>
      <td>78.0</td>
      <td>112.011419</td>
      <td>10.7655</td>
      <td>298.1228</td>
      <td>81.626832</td>
      <td>73.0</td>
      <td>110.659649</td>
      <td>11.5883</td>
      <td>75.5759</td>
      <td>34.411318</td>
      <td>0.0</td>
      <td>128.473076</td>
      <td>11.5883</td>
      <td>75.5759</td>
      <td>34.411318</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.2184</td>
      <td>-11.7815</td>
      <td>46.958254</td>
      <td>38.0</td>
      <td>-35.639534</td>
      <td>-2.4088</td>
      <td>-103.0699</td>
      <td>45.369683</td>
      <td>34.0</td>
      <td>-36.015360</td>
      <td>-0.028919</td>
      <td>0.060370</td>
      <td>-1.695760</td>
      <td>-0.236651</td>
      <td>0.803498</td>
      <td>-0.004446</td>
      <td>-0.008763</td>
      <td>-1.611846</td>
      <td>-0.589330</td>
      <td>0.798413</td>
      <td>-0.061937</td>
      <td>0.060148</td>
      <td>-1.613404</td>
      <td>-0.543836</td>
      <td>0.871159</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6.0107</td>
      <td>677.6633</td>
      <td>77.919512</td>
      <td>116.0</td>
      <td>127.438488</td>
      <td>11.6261</td>
      <td>425.9461</td>
      <td>99.252549</td>
      <td>50.0</td>
      <td>109.396882</td>
      <td>11.0880</td>
      <td>347.3595</td>
      <td>106.446858</td>
      <td>50.0</td>
      <td>102.744033</td>
      <td>9.6115</td>
      <td>37.4252</td>
      <td>22.315708</td>
      <td>0.0</td>
      <td>146.797259</td>
      <td>9.6115</td>
      <td>37.4252</td>
      <td>22.315708</td>
      <td>0.0</td>
      <td>...</td>
      <td>-4.6220</td>
      <td>256.0503</td>
      <td>69.225478</td>
      <td>47.0</td>
      <td>-46.009349</td>
      <td>-9.3363</td>
      <td>140.6224</td>
      <td>69.197065</td>
      <td>47.0</td>
      <td>-48.727307</td>
      <td>-0.659975</td>
      <td>-0.634496</td>
      <td>-1.123664</td>
      <td>0.068966</td>
      <td>0.416575</td>
      <td>-0.340454</td>
      <td>-0.561634</td>
      <td>-0.880947</td>
      <td>-0.500000</td>
      <td>0.489346</td>
      <td>-0.374539</td>
      <td>-0.519589</td>
      <td>-0.864834</td>
      <td>-0.500000</td>
      <td>0.538298</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 181 columns</p>
</div>



## 2-3. Ability Features

`event : ability`의 `event_contents`에는 유닛 및 건물 생산, 업그레이드, 스킬 사용 등 많은 정보가 들어있습니다. 

`re` 라이브러리에서 정규표현식을 작성하여 `event_contents`에서 유닛이나 건물 이름만 추출했습니다. 주로 `Train, Build, Research, Upgrade` 뒤에 특정 단어가 나오는 형태로 저장되어 있었으며, `Morph, WarpIn`처럼 특정 종족에서만 존재하는 명령어도 있습니다.

    train_reg = re.compile(r'Train([A-Z][\w]+)') #Train으로 시작하고, 첫번째 문자가 대문자인 문자열 추출
    data['units'] = data['event_contents'].apply(train_reg)
    
스타크레프트에서는 유닛을 생산하기 위해 특정 건물이나 업그레이드가 필요합니다. 이러한 선행빌드 정보를 이용해서 일부 삭제된 데이터를 복원하려고 했습니다. 예를 들어 다크템플러가 생산했다면 다크템플러의 선행빌드를 확인하고 특정 건물의 기록이 없는경우, 해당 건물의 생산 수를 +1 해주는 형식을 적용하였습니다. 선행빌드 데이터는
[Startcraft2 unit stats](https://liquipedia.net/starcraft2/Terran_Unit_Statistics_(Wings_of_Liberty_and_Heart_of_the_Swarm)
를 참고하여 만들었습니다. 예시는 아래와 같습니다.
    
    p_tech['DarkTemplar'] = ['Gateway', 'CyberneticsCore', 'TwilightCouncil', 'DarkShrine'] #선행 빌드를 저장한 dictionary
    for tech in p_tech['DarkTemplar']: #필요한 선행 빌드의 갯수 확인
        if player[tech] < 0 :
            player[tech] = 1

위의 과정을 통해 만들어진 데이터는 아래와 같습니다. 유닛&건물&업그레이드 이름으로 구성된 columns과 game_id와 player로 구분된 rows를 가지고 있습니다.


```python
player_unit = pd.read_csv('../input/dacon-game/player_unit_train_v2.csv')
player_unit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_id</th>
      <th>player</th>
      <th>SCV</th>
      <th>MULE</th>
      <th>Marine</th>
      <th>Marauder</th>
      <th>Reaper</th>
      <th>Ghost</th>
      <th>Hellion</th>
      <th>BattleHellion</th>
      <th>...</th>
      <th>Blink</th>
      <th>PsiStormTech</th>
      <th>DarkTemplarBlinkUpgrade</th>
      <th>GraviticBoosters</th>
      <th>GraviticDrive</th>
      <th>ExtendedThermalLance</th>
      <th>AnionPulseCrystals</th>
      <th>FluxVanes</th>
      <th>OracleStasisTrap</th>
      <th>Gate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 222 columns</p>
</div>



추출한 데이터와 각 유닛을 만들때 필요한 자원량을 저장한 csv 파일을 연동하여 플레이어가 사용한 자원량의 총합을 구하고 개별 유닛을 특정 그룹으로 묶어 해당 그룹에서 사용한 자원량을 따로 계산하였습니다.

- 총합 관련 변수
    1. `MINERALS`
    2. `GAS`
    3. `Total(Resource)` : MINERALS + GAS
    4. `SUPPLY`
    
    
- 분류한 그룹은 다음과 같으며, 각 그룹에 사용된 자원량('MINERALS', 'GAS' 'SUPPLY')과 전체 자원량에서 해당 그룹의 자원량이 차지하고 있는 비율(`rate`)을 계산했습니다.
    1. `G_unit` : 지상유닛
    2. `A_unit` : 공중유닛
    3. `Upgrade` : 스킬 개발
    4. `Research` : 무기 및 방어구 강화
    5. `Worker` : 일꾼
    6. `Unit` : G_unit + A_unit
    7. `CoreB` : 유닛을 생산할 수 있는 건물(ex. Barracks, Gateway...)
    8. `StatU` : 특수 능력을 사용하는 유닛(ex. HighTemplar, Ghost...)
    9. `Defense` : 방어용 건물(ex. PhotonCannon..)


- 기타 변수
    1. `is_clock` : 투명 유닛 생산 여부
    2. `is_detect` : 탐지 유닛 생산 여부
    3. `core_c` : 종족 별 핵심 스킬(`'ScannerSweep', 'ChronoBoost', 'SpawnLarva'`)의 사용 횟수 
    4. `skill_c` : 스킬 사용 횟수
    5. `drop_c` : 드랍 관련 행동 횟수
    



```python
train_unit = pd.read_csv('../input/dacon-game/train_unit_v5_1.csv')
train_unit[train_unit.columns[:50]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_id</th>
      <th>Worker_Count_0</th>
      <th>Gas_Count_0</th>
      <th>G_unit_MINERALS_0</th>
      <th>G_unit_GAS_0</th>
      <th>G_unit_SUPPLY_0</th>
      <th>G_unit_Count_0</th>
      <th>A_unit_MINERALS_0</th>
      <th>A_unit_GAS_0</th>
      <th>A_unit_SUPPLY_0</th>
      <th>A_unit_Count_0</th>
      <th>Build_MINERALS_0</th>
      <th>Build_GAS_0</th>
      <th>Build_Count_0</th>
      <th>Research_MINERALS_0</th>
      <th>Research_GAS_0</th>
      <th>Research_Count_0</th>
      <th>Upgrade_MINERALS_0</th>
      <th>Upgrade_GAS_0</th>
      <th>Upgrade_Count_0</th>
      <th>core_c_0</th>
      <th>drop_c_0</th>
      <th>skill_c_0</th>
      <th>CoreB_resource_0</th>
      <th>StatU_resource_0</th>
      <th>Defense_resource_0</th>
      <th>Minerals_0</th>
      <th>Gas_0</th>
      <th>SUPPLY_0</th>
      <th>Upgrade_0</th>
      <th>Research_0</th>
      <th>G_unit_0</th>
      <th>A_unit_0</th>
      <th>Build_0</th>
      <th>is_clock_0</th>
      <th>is_detector_0</th>
      <th>Total_0</th>
      <th>Unit_rate_0</th>
      <th>RU_rate_0</th>
      <th>Build_rate_0</th>
      <th>CoreB_rate_0</th>
      <th>StatU_rate_0</th>
      <th>Defense_rate_0</th>
      <th>Worker_Count_1</th>
      <th>Gas_Count_1</th>
      <th>G_unit_MINERALS_1</th>
      <th>G_unit_GAS_1</th>
      <th>G_unit_SUPPLY_1</th>
      <th>G_unit_Count_1</th>
      <th>A_unit_MINERALS_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>21.0</td>
      <td>2.0</td>
      <td>425.0</td>
      <td>75.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2725.0</td>
      <td>275.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>950.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4400.0</td>
      <td>550.0</td>
      <td>32.0</td>
      <td>200.0</td>
      <td>0.0</td>
      <td>500.0</td>
      <td>200.0</td>
      <td>3000.0</td>
      <td>False</td>
      <td>True</td>
      <td>4950.0</td>
      <td>0.141</td>
      <td>0.040</td>
      <td>0.606</td>
      <td>0.192</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>18.0</td>
      <td>2.0</td>
      <td>750.0</td>
      <td>250.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>34.0</td>
      <td>3.0</td>
      <td>625.0</td>
      <td>300.0</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>475.0</td>
      <td>325.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>3225.0</td>
      <td>650.0</td>
      <td>25.0</td>
      <td>250.0</td>
      <td>250.0</td>
      <td>2.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>550.0</td>
      <td>200.0</td>
      <td>0.0</td>
      <td>6375.0</td>
      <td>1625.0</td>
      <td>54.0</td>
      <td>200.0</td>
      <td>500.0</td>
      <td>925.0</td>
      <td>800.0</td>
      <td>3875.0</td>
      <td>False</td>
      <td>True</td>
      <td>8000.0</td>
      <td>0.216</td>
      <td>0.088</td>
      <td>0.484</td>
      <td>0.069</td>
      <td>0.025</td>
      <td>0.000</td>
      <td>29.0</td>
      <td>2.0</td>
      <td>1225.0</td>
      <td>450.0</td>
      <td>25.0</td>
      <td>18.0</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>225.0</td>
      <td>75.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>900.0</td>
      <td>800.0</td>
      <td>17.0</td>
      <td>4.0</td>
      <td>3650.0</td>
      <td>900.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>200.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>700.0</td>
      <td>1000.0</td>
      <td>300.0</td>
      <td>6525.0</td>
      <td>1975.0</td>
      <td>52.0</td>
      <td>400.0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>1700.0</td>
      <td>4550.0</td>
      <td>False</td>
      <td>True</td>
      <td>8500.0</td>
      <td>0.235</td>
      <td>0.047</td>
      <td>0.535</td>
      <td>0.082</td>
      <td>0.118</td>
      <td>0.035</td>
      <td>35.0</td>
      <td>3.0</td>
      <td>750.0</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>800.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>19.0</td>
      <td>1.0</td>
      <td>1825.0</td>
      <td>450.0</td>
      <td>37.0</td>
      <td>27.0</td>
      <td>450.0</td>
      <td>450.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>3250.0</td>
      <td>375.0</td>
      <td>22.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>1.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>950.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6675.0</td>
      <td>1475.0</td>
      <td>65.0</td>
      <td>200.0</td>
      <td>200.0</td>
      <td>2275.0</td>
      <td>900.0</td>
      <td>3625.0</td>
      <td>False</td>
      <td>True</td>
      <td>8150.0</td>
      <td>0.390</td>
      <td>0.049</td>
      <td>0.445</td>
      <td>0.117</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>36.0</td>
      <td>4.0</td>
      <td>1750.0</td>
      <td>700.0</td>
      <td>28.0</td>
      <td>13.0</td>
      <td>375.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>26.0</td>
      <td>2.0</td>
      <td>250.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>150.0</td>
      <td>100.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1900.0</td>
      <td>300.0</td>
      <td>13.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>650.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3700.0</td>
      <td>500.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>250.0</td>
      <td>250.0</td>
      <td>2200.0</td>
      <td>True</td>
      <td>True</td>
      <td>4200.0</td>
      <td>0.119</td>
      <td>0.048</td>
      <td>0.524</td>
      <td>0.155</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>325.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>500.0</td>
    </tr>
  </tbody>
</table>
</div>



## Modeling

모델은 lightGBM을 사용했습니다.

## Feature selection

features의 숫자를 줄이기 위해 여러 방법을 사용했습니다.

1. correlation features selection
독립 변수 간의 상관계수를 구하고 특정 수치보다 높은 상관계수 값을 보이면 제거하였습니다. 설정한 threshould 값은 0.995이며, 총 xx개의 변수를 삭제하였습니다.

2. Permutation importance
특정 변수의 값을 무작위로 섞어버린후(shuffling), 모델의 score을 계산합니다. 원래의 score보다 값이 상승했다면, 해당 변수는 의미없는 변수라고 판단하게 됩니다. 해당 방법을 적용해서 총 xx개의 변수를 삭제했습니다.

## Fold selection

1. `kfold` 
5 fold로 나눠서 학습을 진행했으며, `shuffle=True`로 무작위로 섞은 학습 데이터를 사용했습니다.

2. `stratified Kfold` by `species`
`species`를 기준으로 stratified Kfold를 적용했습니다.

3. `stratified Kfold` by `max_time`
게임의 시간을 분 단위로 분류해서 총 5개의 category로 만든 `max_time`를 기준으로 stratified Kfold를 적용했습니다.

4. 따로 학습 
종족 간 모델을 구별해서 학습을 시켰습니다. 
    1. 테란이 포함된 게임
    2. 프로토스가 포함된 게임
    3. 저그가 포함된 게임
    

## 승자 코드 분석

1. 시간 및 갯수 등을 제외하면 단순히 행동 코드에 따라 `dummy` 형태로 변수를 생성함
2. 종족별로 따로 모델을 학습함
