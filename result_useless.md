# 1. Save Experiment Result


# 2. JPSC1400 Use CNN

## 2.1. 7fontJpan padding=2 比较SSCD生成算法

Base : CC-GF-RF-GF-RF-GF-RF-GF-GN
Base : GF-RF-GF-RF-GF-RF-GF-GN

### 500000 + MyNet44(no33)(DLD) - 40 epoches 

MO(Horie)-PT_dir(-5,5)-CC-GF-RF-GN
Finally top1 accuracy: 0.8121428571428572
Finally top2 accuracy: 0.8757142857142857
Best : [Epoch: 27] --- Iteration: 105300, Acc: 0.835.


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC-GF-RF-GN
Finally top1 accuracy: 0.7914285714285715
Finally top2 accuracy: 0.8671428571428571
Best : [Epoch: 39] --- Iteration: 150000, Acc: 0.8185714285714286.


----------------------------------------------------------
MO(0.7,d4,d8,pU,pD)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8328571428571429
Finally top2 accuracy: 0.9035714285714286
Best : [Epoch: 30] --- Iteration: 116100, Acc: 0.8507142857142858


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8321428571428572
Finally top2 accuracy: 0.8985714285714286
Best : [Epoch: 36] --- Iteration: 139800, Acc: 0.8585714285714285.


MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8314285714285714
Finally top2 accuracy: 0.895
Best : [Epoch: 33] --- Iteration: 127900, Acc: 0.8435714285714285.


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.835
Finally top2 accuracy: 0.9007142857142857
Best : [Epoch: 40] --- Iteration: 154700, Acc: 0.855

----------------------------------------------------------------


### 500000 + MyVGG(no33)(DLD) - 40 epoches 

MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8378571428571429
Finally top2 accuracy: 0.8978571428571429
Best : [Epoch: 37] --- Iteration: 141100, Acc: 0.8528571428571429.




### 2.1.1. 500000 + MyNet44(no33)(DLD)  - LOSS 0.5 (废弃)

MyNet44-512:
MO(0.7)-PT(-0.1,0.2,p=1)-Base: 

Finally correct predicted: 0.8271428571428572
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 62900, Acc: 0.8442857142857143.
[Epoch: 19] --- Iteration: 74233, Loss: 0.47950335680459494.

Finally correct predicted: 0.8521428571428571
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 69000, Acc: 0.8614285714285714.
[Epoch: 20] --- Iteration: 78140, Loss: 0.4976906088335422

Finally correct predicted: 0.8357142857142857
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 70600, Acc: 0.8478571428571429.
[Epoch: 22] --- Iteration: 85954, Loss: 0.4985309371130388.

Finally correct predicted: 0.8278571428571428
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 72200, Acc: 0.8414285714285714.
[Epoch: 19] --- Iteration: 74233, Loss: 0.47821108121250566.

--------------------------------------------

MyNet44-512:
MO(0.7)-PT(-0.1,0.2,p=1)-AT(p=1,shear(15,15),autoscale)-Base: 
Finally correct predicted: 0.8321428571428572
Predict one set cost time: 2 s
Best : [Epoch: 31] --- Iteration: 118500, Acc: 0.8485714285714285.
[Epoch: 35] --- Iteration: 136745, Loss: 0.49174044895602764.

--------------------------------------------

MyNet44-768:
MO(0.7)-PT(-0.1,0.2,p=1)-Base: 

Finally correct predicted: 0.8357142857142857
Predict one set cost time: 2 s
Best : [Epoch: 14] --- Iteration: 54100, Acc: 0.8571428571428571.
[Epoch: 20] --- Iteration: 78140, Loss: 0.48952570915440635.

### 2.1.2. 500000 + MyVGG （废弃）

MO(0.7)-PT(-0.1,0.2,p=1)-Base  
Finally correct predicted: 0.8264285714285714
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38300, Acc: 0.8428571428571429.

MO(0.3)-AF(ro=30,sh=15,p=1,autoscale)-Base


MO(0.3)-AF(ro=20,sh=15,p=1,autoscale)-Base


### 2.1.3. (废弃) 500000 + 50次单bunch  + MyNet 44-512 （with first 33)

Train cost: 2 h 40 m 40 s
Train cost: 2 h 42 m 27 s
Train cost: 2 h 44 m 43 s 
Train cost: 2 h 45 m 10 s 


- MO 概率 0.3----------------------------------------------------------------------
- AF ro=30 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.3)-AF(ro=30,sh=15,p=0.7)-PT(-0.1,0.2,p=1)-Base
Final : 0.7985714285714286
Best  : [Epoch: 21] --- Iteration: 100, Acc: 0.8185714285714286

MO(0.3)-AF(ro=30,sh=15,p=1)-PT(-0.1,0.2,p=1)-Base
Final :0.7978571428571428
Best  : [Epoch: 58] --- Iteration: 100, Acc: 0.7978571428571428

- MO 概率 0.3---------------------------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变
  
MO(0.3)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base
Final : 0.8285714285714286
Best  : [Epoch: 20] --- Iteration: 100, Acc: 0.8457142857142858
MobileNet: 0.8128571428571428
MobileNet Bset : 0.8278571428571428

MO(0.3)-AF(ro=20,sh=15,p=1)-PT(-0.1,0.2,p=1)-Base
Final : 0.8307142857142857
Best  : [Epoch: 36] --- Iteration: 100, Acc: 0.8385714285714285


- MO 概率 0.3 调换AF PT 的顺序  ------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变
MO(0.3)-PT(-0.1,0.2,p=1)-AF(ro=20,sh=15,p=0.5)-Base
Final : 0.8385714285714285
Best  : [Epoch: 28] --- Iteration: 100, Acc: 0.8414285714285714.

MO(0.3)-PT(-0.1,0.2,p=1)-AF(ro=20,sh=15,p=1)-Base
Final : 0.7735714285714286
Best  : [Epoch: 52] --- Iteration: 100, Acc: 0.81


- MO 的概率上升到 0.7 -------------------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.7)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base
Final : 0.8335714285714285
Best  : [Epoch: 47] --- Iteration: 100, Acc: 0.8464285714285714

MO(0.7)-AF(ro=20,sh=15,p=0.7)-PT(-0.1,0.2,p=1)-Base
Final : 0.8128571428571428
Best  : [Epoch: 42] --- Iteration: 100, Acc: 0.8235714285714286


- MO 的概率上升到 0.7 调换AF PT 的顺序  -------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.7)-PT(-0.1,0.2,p=1)--AF(ro=20,sh=15,p=1)Base
Final : 0.8371428571428572
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.835

MO(0.7)-PT(-0.1,0.2,p=1)--AF(ro=20,sh=15,p=0.7)Base
Final : 0.8292857142857143
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8407142857142857



## 2.2. 7fontJpan padding=2 比较网络结构 (训练方法已废弃)

SSCD 都是 MO(0.7,d4,d8,pU)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base

* 标题是训练数据量和方法
* 这里的 MyNet 默认是带 first33 的

### 2.2.1. 500k(adam,0.0001,e=10)

- MyNet42(512)
Finally correct predicted: 0.76
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37000, Acc: 0.7742857142857142.
[Epoch: 10] --- Iteration: 39000, Loss: 3.704668809183133.

- MyNet42(512, no33)
Finally correct predicted: 0.7585714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38700, Acc: 0.77.
[Epoch: 10] --- Iteration: 39000, Loss: 2.662528754743246

------------------------------------------------------------------------

- MyNet44(512)
Finally correct predicted: 0.815
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37200, Acc: 0.8278571428571428
[Epoch: 10] --- Iteration: 39000, Loss: 3.202185431713764

- MyNet44(512, no33)
Finally correct predicted: 0.8135714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 35500, Acc: 0.8342857142857143.
[Epoch: 10] --- Iteration: 39000, Loss: 1.7207405022236113.

- MyNet44(512, no33,LD-L)
Finally correct predicted: 0.8207142857142857
Predict one set cost time: 2 s
Best : [Epoch: 8] --- Iteration: 27400, Acc: 0.8421428571428572.
[Epoch: 10] --- Iteration: 39000, Loss: 0.9498902468694858.

- MyNet44(512, no33,LL-L)
Finally correct predicted: 0.7735714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 36400, Acc: 0.7942857142857143
[Epoch: 10] --- Iteration: 39000, Loss: 1.5581608724384808

- MyNet44(512, no33,LLD-L)
Finally correct predicted: 0.7985714285714286
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 31500, Acc: 0.82.
[Epoch: 10] --- Iteration: 39000, Loss: 1.821282786067605


- MyVGG41(512)
Finally correct predicted: 0.82
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 32800, Acc: 0.83.
[Epoch: 10] --- Iteration: 39000, Loss: 2.2967642378851965

------------------------------------------------------------------------


- MyNet46(512)
Finally correct predicted: 0.7942857142857143
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37600, Acc: 0.8085714285714286.
[Epoch: 10] --- Iteration: 39000, Loss: 3.563677761330914

- MyNet46(512, no33)
Finally correct predicted: 0.8
Predict one set cost time: 2 s
Best : [Epoch: 7] --- Iteration: 27000, Acc: 0.8171428571428572
[Epoch: 10] --- Iteration: 39000, Loss: 1.732441489480197.


- MyNet44-4(512,no33)
Finally correct predicted: 0.7107142857142857
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37400, Acc: 0.72.
[Epoch: 10] --- Iteration: 39000, Loss: 3.538749488861133.

- MyNet44-4(512)
Finally correct predicted: 0.775
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37200, Acc: 0.7835714285714286
[Epoch: 10] --- Iteration: 39000, Loss: 2.160151597893773.


- MyNet46-4(512,no33)
Finally correct predicted: 0.7578571428571429
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 34800, Acc: 0.7757142857142857.
[Epoch: 10] --- Iteration: 39000, Loss: 2.0203742220741816.

- MyNet46-4(512)
Finally correct predicted: 0.7585714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38600, Acc: 0.7714285714285715
[Epoch: 10] --- Iteration: 39000, Loss: 3.484916457338211.



### 2.2.2. 500k(adam,0.0001,e=10) + 10*0.500k(adam,0.0001)

- MyNet44(512)
Final: 0.8157142857142857
Best : [Epoch: 20] --- Iteration: 300, Acc: 0.8364285714285714.

-MyNet44(384)
Final : 0.8285714285714286
Best  : [Epoch: 20] --- Iteration: 200, Acc: 0.8278571428571428


### 2.2.3. 500k(adam,0.0001,e=10) + 10*0.500k(adam,0.001,lineral)
- MyNet42(512)
Finally correct predicted: 0.8057142857142857
Predict one set cost time: 2 s
Best : [Epoch: 52] --- Acc: 0.8128571428571428


### 2.2.4. 500k(adam,0.0001,e=10)-20*0.500k(adam,0.001,lineral)

- MyVGG41(512)
Finally correct predicted: 0.835
Predict one set cost time: 2 s
Best : [Epoch: 28] --- Iteration: 200, Acc: 0.8407142857142857


### 2.2.5. 500k(adam,0.0001,e=10)-50*1bun(adam,0.0001)

- MyNet44(384)
Final : 0.8285714285714286
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8428571428571429

- MyNet44(512)
Final : 0.8335714285714285
Best  : [Epoch: 47] --- Iteration: 100, Acc: 0.8464285714285714

- MyNet46(512)
Final : 0.8278571428571428
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8271428571428572


### 2.2.6. 500k(adam,0.0001,e=10)-50*1bun(adam,0.001,lineral)

- MyVGG41(512)
Finally correct predicted: 0.8321428571428572
Predict one set cost time: 2 s
Best : [Epoch: 55] --- Acc: 0.8421428571428572.

- MyNet42(512)
Finally correct predicted: 0.8057142857142857
Predict one set cost time: 2 s
Best : [Epoch: 52] --- Acc: 0.8128571428571428

## 2.3. 网络结构专门对比

* MO-PT-CC-GF-RF-GF-RF-GF-RF-GF-GN 
* 这里的 MyNet 都不再使用 first 33

### 2.3.1. (废弃) 500k + 10 epoch  发现 10 epoch 还不足够充分收敛 
7font

MyNet42
best:     0.7757142857142857
bestiter: 36000
final:    0.7714285714285715  

MyNet44
best:     0.8278571428571428
bestiter: 25700
final:    0.8157142857142857

MyNet46:
best:     0.8171428571428572
bestiter: 37900
final:    0.8092857142857143

MyNet48:
best:     0.8057142857142857
bestiter: 35500
final:    0.7892857142857143


MobileNetV2:  
Finally correct predicted: 0.7928571428571428
Best : [Epoch: 10] --- Iteration: 38800, Acc: 0.8028571428571428

MyVGG
Finally correct predicted: 0.8264285714285714
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38300, Acc: 0.8428571428571429

ResNet34
Finally correct predicted: 0.8228571428571428
Predict one set cost time: 4 s
Best : [Epoch: 10] --- Iteration: 35800, Acc: 0.8435714285714285

-----------------------------------------------------------------------------------

9font 加入自搜的2个手写字体
MO-PT-CC-GF-RF-GF-RF-GF-RF-GF-GN  
500k + 10 epoch  
| 类型网络 | 2                  | 4                  | 6                  | 8                  | vgg                |
| -------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| best     | 0.8057142857142857 | 0.855              | 0.845              | 0.8292857142857143 | 0.855              |
| bestiter | 37800              | 37300              | 30200              | 38300              | 35700              |
| final    | 0.8007142857142857 | 0.8407142857142857 | 0.8314285714285714 | 0.815              | 0.8307142857142857 |

### 2.3.2. (废弃) 500k + 20 epoch 发现20 epoch 仍然有提升空间

MyNet42(LDLD)
Finally correct predicted: 0.7907142857142857
Predict one set cost time: 2 s
Best : [Epoch: 20] --- Iteration: 78000, Acc: 0.7964285714285714. 

MyNet43(LDLD)
Finally correct predicted: 0.8035714285714286
Predict one set cost time: 2 s
Best : [Epoch: 14] --- Iteration: 52000, Acc: 0.8292857142857143

MyNet43(D)
Finally correct predicted: 0.8164285714285714
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 69900, Acc: 0.8285714285714286

MyNet44(LDLD)
Finally correct predicted: 0.8292857142857143
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 71200, Acc: 0.845.

MyNet44(D)
Finally correct predicted: 0.825
Predict one set cost time: 2 s  
Best : [Epoch: 17] --- Iteration: 65300, Acc: 0.8471428571428572.

MyNet44(DLD)
Finally correct predicted: 0.85
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 62900, Acc: 0.8585714285714285.
Finally correct predicted: 0.8442857142857143
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 65900, Acc: 0.8514285714285714.

MyNet45(LDLD)
Finally correct predicted: 0.8042857142857143
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 64100, Acc: 0.8257142857142857

MyNet45(D)
Finally correct predicted: 0.8435714285714285
Predict one set cost time: 2 s
Best : [Epoch: 16] --- Iteration: 60800, Acc: 0.85

MyNet45(DLD)
Finally correct predicted: 0.84
Predict one set cost time: 2 s
Best : [Epoch: 20] --- Iteration: 77700, Acc: 0.8542857142857143.

MyNet46(DLD)
Finally correct predicted: 0.8292857142857143
Predict one set cost time: 2 s
Best : [Epoch: 16] --- Iteration: 59600, Acc: 0.8528571428571429

MyNet46(LDLD)
Finally correct predicted: 0.8021428571428572
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 66900, Acc: 0.8164285714285714


MyNet47(DLD)
Finally correct predicted: 0.8257142857142857
Predict one set cost time: 2 s
Best : [Epoch: 15] --- Iteration: 55100, Acc: 0.855.

MyNet4(6543)(DLD)
Finally correct predicted: 0.8242857142857143 
Predict one set cost time: 2 s 
2.4. Best : [Epoch: 20] --- Iteration: 75600, Acc: 0.8492857142857143.

-----------------------------------------
MyVGG(LDLD)
Finally correct predicted: 0.8285714285714286
Predict one set cost time: 2 s
Best : [Epoch: 15] --- Iteration: 56300, Acc: 0.8478571428571429.

MyVGG(D)
Finally correct predicted: 0.83
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 71100, Acc: 0.845.

------------------------------------------------

MobileNetV2
Finally correct predicted: 0.815
Predict one set cost time: 3 s
Best : [Epoch: 20] --- Iteration: 76100, Acc: 0.8442857142857143.

ResNet34
Finally correct predicted: 0.8357142857142857
Predict one set cost time: 3 s
Best : [Epoch: 14] --- Iteration: 54200, Acc: 0.8464285714285714

### 2.3.3. 500k + 40 epoch

MyVGG(DLD)
Finally top1 accuracy: 0.8621428571428571
Finally top2 accuracy: 0.9092857142857143
Best : [Epoch: 25] --- Iteration: 96700, Acc: 0.8742857142857143.

MyNet44(DLD)
Finally top1 accuracy: 0.8478571428571429
Finally top2 accuracy: 0.9042857142857142
Best : [Epoch: 31] --- Iteration: 120400, Acc: 0.86. 

MyNet43(DLD)
Finally top1 accuracy: 0.8464285714285714
Finally top2 accuracy: 0.9057142857142857
Best : [Epoch: 35] --- Iteration: 136200, Acc: 0.8571428571428571.

MyNet45(DLD)
Finally top1 accuracy: 0.8521428571428571
Finally top2 accuracy: 0.9042857142857142
Best : [Epoch: 39] --- Iteration: 151900, Acc: 0.8592857142857143

MyNet46(DLD)
Finally top1 accuracy: 0.861
Best: 0.869



### 2.3.4. 500K + 相同训练loss

# 3. JPSC1400 Use NNS 120

* 噪点非常重要
* HOG对模糊不敏感
* 随机滤波之后再加噪点效果更好
* 16对噪点非常敏感, GF放在最后对16的精度有提升, 但是对全局精度没好处

* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   

## 3.1. 7fontJpan padding=0

### 3.1.1. NNS

16 : 0.5121428571428571
32 : 0.5414285714285715
64 : 0.5342857142857143

## 3.2. 7fontJpan padding=1
16 : 0.5192857142857142
32 : 0.5371428571428571
64 : 0.5585714285714286
### 3.2.1. NNS

## 3.3. 7fontJpan padding=2


16 : 0.5078571428571429
32 : 0.5257142857142857
64 : 0.5692857142857143

### 3.3.1. 和 Horie 手法比较 种子归一后的数据

CC-GF-RF-GN (种子设定一致)
* 16均值:       0.4899285714285713
* 16ens :       0.6264285714285714
* 32均值 :      0.5903035714285715
* 32ens :       0.665
* 64均值 :      0.561625
* 64ens :       0.6742857142857143
* 120 MSR-3 :   0.7171428571428572

CC(lib20)-GF-RF-GN  (种子设定一致)
* 16均值:       0.5327321428571429 
* 16ens :       0.6507142857142857
* 32均值 :      0.5862499999999998
* 32ens :       0.66
* 64均值 :      0.5587500000000001
* 64ens :       0.6728571428571428
* 120 MSR-3 :   0.7278571428571429

-----------------------------

MO(horie)-CC-GF-RF-GN                   (种子设定一致)
* 16均值:       0.4791964285714285
* 16ens :       0.6521428571428571
* 32均值 :      0.5881607142857144
* 32ens :       0.6985714285714286
* 64均值 :      0.5449285714285714
* 64ens :       0.675
* 120 MSR-3 :   0.7342857142857143

<!-- 
MO(0.7,d4,d8,pU,pD)-CC-GF-RF-GN         (种子设定一致)
* 16均值:       0.4729107142857144
* 16ens :       0.6435714285714286
* 32均值 :      0.5964107142857145
* 32ens :       0.6892857142857143
* 64均值 :      0.5581964285714286
* 64ens :       0.6735714285714286
* 120 MSR-3 :   0.7407142857142858 -->

MO(horie)-CC(lib)-GF-RF-GN              (种子设定一致)
* 16均值:       0.5213035714285715
* 16ens :       0.6742857142857143
* 32均值 :      0.5852678571428569
* 32ens :       0.6914285714285714
* 64均值 :      0.5481964285714286
* 64ens :       0.6842857142857143
* 120 MSR-3 :   0.7414285714285714


MO(0.7,d4,d8,pU,pD)-CC(lib)-GF-RF-GN   (种子设定一致)
* 16均值:       0.5193392857142858
* 16ens :       0.66
* 32均值 :      0.5911607142857144
* 32ens :       0.6878571428571428
* 64均值 :      0.5555892857142857
* 64ens :       0.6764285714285714
* 120 MSR-3 :   0.7364285714285714


MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN   (种子设定一致)
* 16均值:       0.47167857142857156
* 16ens :       0.6471428571428571
* 32均值 :      0.5979107142857143
* 32ens :       0.6928571428571428
* 64均值 :      0.5573392857142857
* 64ens :       0.6678571428571428
* 120 MSR-3 :   0.7342857142857143

----------------------------------------------------------------

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GN   (种子设定一致)
* 16均值:       0.44885714285714295
* 16ens :       0.6614285714285715
* 32均值 :      0.5804107142857142
* 32ens :       0.7085714285714285
* 64均值 :      0.5351964285714284
* 64ens :       0.6878571428571428
* 120 MSR-3 :   0.7471428571428571

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.42716071428571434
* 16ens :       0.6471428571428571
* 32均值 :      0.5867321428571429
* 32ens :       0.7157142857142857
* 64均值 :      0.5428392857142856
* 64ens :       0.7
* 120 MSR-3 :   0.7528571428571429

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GN   (种子设定一致)
* 16均值:       0.4278928571428571
* 16ens :       0.6628571428571428
* 32均值 :      0.5610535714285715
* 32ens :       0.7128571428571429
* 64均值 :      0.5139285714285712
* 64ens :       0.7028571428571428
* 120 MSR-3 :   0.7542857142857143

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN      (种子设定一致)
* 16均值:       0.41089285714285706
* 16ens :       0.6557142857142857
* 32均值 :      0.5640892857142858
* 32ens :       0.7207142857142858
* 64均值 :      0.5198392857142857
* 64ens :       0.7092857142857143
* 120 MSR-3 :   0.7685714285714286

MO(0.7,d4,d8,pU,pD)-PT(dir,-5,5)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.34992857142857153
* 16ens :       0.6514285714285715
* 32均值 :      0.4982678571428571
* 32ens :       0.7414285714285714
* 64均值 :      0.45125000000000004
* 64ens :       0.7257142857142858
* 120 MSR-3 :   0.7807142857142857

MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.3835535714285713
* 16ens :       0.69
* 32均值 :      0.48410714285714285
* 32ens :       0.735
* 64均值 :      0.43892857142857145
* 64ens :       0.7292857142857143
* 120 MSR-3 :   0.7935714285714286


* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   

* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   


--------------------------------------------------------------------------

PT(-0.1,0.2)-CC-GF-RF-GN
* 16均值:     0.39180357142857136
* 16ens :     0.6435714285714286
* 32均值 :    0.5076785714285714
* 32ens :     0.6971428571428572
* 64均值 :    0.47101785714285727
* 64ens :     0.7114285714285714
* 120 MSR-3 : 0.7621428571428571

PT_dir(-5,5)-CC-GF-RF-GN
* 16均值:     0.4165
* 16ens :     0.6564285714285715
* 32均值 :    0.5364464285714287
* 32ens :     0.7185714285714285
* 64均值 :    0.48914285714285705
* 64ens :     0.725
* 120 MSR-3 : 0.7635714285714286



------------------------------------------------
PT_dir(-5,5)-MO(horie)-CC(lib20)-GF-RF-GN
* 16均值:     0.4498214285714286
* 16ens :     0.7107142857142857
* 32均值 :     0.5159464285714284
* 32ens :     0.7292857142857143
* 64均值 :    0.47285714285714275
* 64ens :     0.7214285714285714
* 120 MSR-3 : 0.7771428571428571


PT(-0.1,0.2)-MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN
* 16均值:     0.32730357142857147
* 16ens :     0.6707142857142857
* 32均值 :    0.4996607142857143
* 32ens :     0.725
* 64均值 :    0.4710357142857142
* 64ens :     0.7278571428571429
* 120 MSR-3 : 0.7828571428571428  

Reversed: 
MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.376107142857143
* 16ens :     0.6535714285714286
* 32均值 :    0.5099464285714286
* 32ens :     0.705
* 64均值 :    0.47401785714285716
* 64ens :     0.7285714285714285
* 120 MSR-3 : 0.7664285714285715
------------------------------------------------------


MO(0.7,d4,d8,pU,pD)-PT_dir(-5,5)-CC(gap:50)-GF-RF-GN
* 16均值:     0.3997500000000001
* 16ens :     0.6735714285714286
* 32均值 :    0.5379821428571427
* 32ens :     0.7242857142857143
* 64均值 :    0.4885357142857143
* 64ens :     0.7228571428571429
* 120 MSR-3 : 0.7735714285714286

MO(horie)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.3756785714285714
* 16ens :     0.66
* 32均值 :    0.49557142857142866
* 32ens :     0.7214285714285714
* 64均值 :    0.4543750000000001
* 64ens :     0.7242857142857143
* 120 MSR-3 : 0.7721428571428571

MO(horie)-PT_dir(-5,5)-CC(gap:50)-GF-RF-GN
* 16均值:     0.40514285714285714
* 16ens :     0.6885714285714286
* 32均值 :    0.5240714285714286
* 32ens :     0.7278571428571429
* 64均值 :    0.47648214285714274
* 64ens :     0.73
* 120 MSR-3 : 0.7778571428571428

### 3.3.2. 少变换

CC
* 16均值: 0.5082321428571429
* 16ens :  0.5085714285714286
* 32均值 : 0.5293928571428571
* 32ens :  0.5307142857142857
* 64均值 : 0.5705178571428572
* 64ens : 0.5757142857142857
* 120 MSR-3 : 0.5821428571428572

CC-GN
* 16均值: 0.5229821428571428
* 16ens : 0.5835714285714285
* 32均值 : 0.5901071428571428
* 32ens : 0.6121428571428571
* 64均值 : 0.5714821428571428
* 64ens : 0.6142857142857143
* 120 MSR-3 : 0.6607142857142857

CC-RF-GN
* 16均值: 0.5163749999999998
* 16ens : 0.6107142857142858
* 32均值 : 0.5778392857142857
* 32ens : 0.6442857142857142
* 64均值 : 0.5580178571428569
* 64ens : 0.6564285714285715
* 120 MSR-3 : 0.705

### 3.3.3. 四标准变化单RF

CC-GF-RF-GN
* 16均值: 0.49169642857142853
* 16ens : 0.6307142857142857
* 32均值 : 0.5879107142857142
* 32ens : 0.66
* 64均值 : 0.5593749999999998
* 64ens : 0.67
* 120 MSR-3 : 0.7235714285714285
* 120 MSR-3 : 0.7228571428571429 (第二次)

CC(lib20)-GF-RF-GN
* 16均值: 0.5299285714285713
* 16ens : 0.6464285714285715
* 32均值 : 0.5845178571428572
* 32ens : 0.6685714285714286
* 64均值 : 0.557392857142857
* 64ens : 0.6728571428571428
* 120 MSR-3 : 0.7185714285714285

CC-GF-RF-GN-GF
* 16均值: 0.5853750000000001
* 16ens : 0.6528571428571428
* 32均值 : 0.6016250000000001
* 32ens : 0.6664285714285715
* 64均值 : 0.5709107142857145
* 64ens : 0.6764285714285714
* 120 MSR-3 : 0.7214285714285714  反而还降了，

CC-GF-RF-GF-GN
* 16均值: 0.4588214285714285
* 16ens : 0.6235714285714286
* 32均值 : 0.5991428571428573
* 32ens : 0.6778571428571428
* 64均值 : 0.5685535714285714
* 64ens : 0.685
* 120 MSR-3 : 0.7321428571428571

CC-GF-RF-GN-GF-GN-GF
* 16均值:  0.5751071428571428
* 16ens : 0.6664285714285715
* 32均值 : 0.6082499999999998
* 32ens : 0.6864285714285714
* 64均值 : 0.5711964285714288
* 64ens : 0.685
* 120 MSR-3 : 0.7271428571428571


### 3.3.4. 四标准多RF

CC-GF-RF-GF-RF-GF-RF-GF-GN

CC-GF-RF-GF-RF-GF-GN
* 16均值: 0.4432142857142859
* 16ens : 0.6521428571428571
* 32均值 : 0.5862142857142858
* 32ens : 0.7157142857142857
* 64均值 : 0.5513571428571428
* 64ens : 0.7028571428571428
* 120 MSR-3 :  0.7535714285714286

CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.4226964285714286
* 16ens : 0.655
* 32均值 : 0.56625
* 32ens : 0.7192857142857143
* 64均值 : 0.5238571428571428
* 64ens :  0.7078571428571429 
* 120 MSR-3 : 0.7578571428571429

CC-GF-RF-GF-RF-GF-RF-GF-GN (2)
* 16均值: 0.42312500000000003
* 16ens : 0.6628571428571428
* 32均值 : 0.5685714285714285
* 32ens : 0.7207142857142858
* 64均值 : 0.528357142857143
* 64ens : 0.71
* 120 MSR-3 : 0.7621428571428571



### 3.3.5. 加入单几何或形态学

-----------------------MO-------------------------

MO(p=0.3, d4,d8,pU)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.42207142857142854
* 16ens : 0.6714285714285714
* 32均值 : 0.5697321428571427
* 32ens : 0.7271428571428571
* 64均值 : 0.523857142857143
* 64ens : 0.7085714285714285
* 120 MSR-3 : 0.7685714285714286

MO(p=0.3, d4,d8,pU)-CC-GF-RF-GF-RF-GF-RF-GF-GN (2)
* 16均值: 0.42207142857142854
* 16ens : 0.6714285714285714
* 32均值 : 0.5697321428571427
* 32ens : 0.7271428571428571
* 64均值 : 0.523857142857143
* 64ens : 0.7085714285714285
* 120 MSR-3 : 0.7685714285714286

---------------------------AF --------------------------------

AF(p=1, ro=20, sh=(10,10))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.28858928571428577
* 16ens : 0.6257142857142857
* 32均值 : 0.4144821428571429
* 32ens : 0.6992857142857143
* 64均值 : 0.373482142857143
* 64ens : 0.685
* 120 MSR-3 : 0.755

 
AF(p=1, ro=20, sh=(15,15))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.27005357142857145
* 16ens : 0.6264285714285714
* 32均值 : 0.3912321428571428
* 32ens : 0.695
* 64均值 : 0.35505357142857147
* 64ens : 0.6971428571428572
* 120 MSR-3 : 0.7592857142857142

AF(p=1, ro=30, sh=(10,10))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.24335714285714288
* 16ens : 0.5835714285714285
* 32均值 : 0.36321428571428577
* 32ens : 0.6907142857142857
* 64均值 : 0.31519642857142854
* 64ens :  0.67
* 120 MSR-3 : 0.7464285714285714

AF(p=1, ro=30, sh=(15,15))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.22530357142857146
* 16ens : 0.5657142857142857
* 32均值 : 0.33825
* 32ens :  0.6578571428571428
* 64均值 : 0.30189285714285713
* 64ens :  0.6621428571428571
* 120 MSR-3 : 0.7321428571428571

-----------------------------------------PT--------------------------

PT(0,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.38489285714285715
* 16ens : 0.665
* 32均值 : 0.5291428571428571
* 32ens : 0.7135714285714285
* 64均值 : 0.5013928571428572
* 64ens :  0.7114285714285714
* 120 MSR-3 : 0.7685714285714286

PT(0,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3215892857142858
* 16ens : 0.615 
* 32均值 : 0.45326785714285717
* 32ens : 0.6921428571428572
* 64均值 : 0.42474999999999985
* 64ens : 0.6764285714285714
* 120 MSR-3 : 0.745

PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3964285714285714
* 16ens : 0.6664285714285715
* 32均值 : 0.5350535714285714
* 32ens : 0.7228571428571429
* 64均值 : 0.4935892857142858
* 64ens : 0.7278571428571429
* 120 MSR-3 : 0.7742857142857142

PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3322142857142857
* 16ens : 0.6571428571428571
* 32均值 : 0.47275
* 32ens : 0.7228571428571429
* 64均值 : 0.434875
* 64ens :  0.7128571428571429
* 120 MSR-3 : 0.77
 
### 3.3.6. 形态学+投影变换


MO(p=0.3, d4,d8,pU)-PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :

MO(p=0.7, d4,d8,pU)-PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.38658928571428575
* 16ens : 0.6757142857142857
* 32均值 : 0.5393035714285717
* 32ens : 0.7314285714285714
* 64均值 : 0.48953571428571435
* 64ens :  0.7157142857142857
* 120 MSR-3 : 0.7792857142857142


MO(p=0.3, d4,d8,pU)-PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.32866071428571425
* 16ens : 0.6678571428571428
* 32均值 : 0.4697321428571429
* 32ens : 0.7307142857142858
* 64均值 : 0.4373214285714284
* 64ens :  0.7292857142857143
* 120 MSR-3 : 0.7821428571428571

MO(p=0.7, d4,d8,pU)-PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.3263214285714286
* 16ens :  0.6578571428571428
* 32均值 : 0.4781607142857144
* 32ens :  0.735
* 64均值 : 0.43808928571428574
* 64ens :  0.7171428571428572
* 120 MSR-3 : 0.7821428571428571

--------------------------------------------

PT(-0.1,0.2)-MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN
* 16均值:     0.32730357142857147
* 16ens :     0.6707142857142857
* 32均值 :    0.4996607142857143
* 32ens :     0.725
* 64均值 :    0.4710357142857142
* 64ens :     0.7278571428571429
* 120 MSR-3 : 0.7828571428571428  

Reversed: 
MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.376107142857143
* 16ens :     0.6535714285714286
* 32均值 :    0.5099464285714286
* 32ens :     0.705
* 64均值 :    0.47401785714285716
* 64ens :     0.7285714285714285
* 120 MSR-3 : 0.7664285714285715



## 3.4. 7fontJpan revised  3097

## 3.5. 7fontJpan padding=2 revised  3104　ヘぺべ　已舍弃 

### 3.5.1. NNS

* 16 ：0.51
* 32 ：0.5278571428571428
* 64 ：0.5707142857142857

* BaseLine : 57



#### 3.5.1.1. 单变换 baseline

颜色 gap 50:
* 16均值:   0.5107857142857143
* 16ens :  0.5107142857142857
* 32均值 : 0.5310535714285713
* 32ens :  0.5328571428571428
* 64均值 :  0.5730357142857144
* 64ens :  0.575
* 120 MSR-3 : 0.5707142857142857

#### 3.5.1.2. 双变换 

Color(gap:50)-GF(sigma: 0-10)
* 16均值: 0.5099107142857142
* 16ens : 0.515
* 32均值 : 0.5586785714285714 
* 32ens : 0.5607142857142857
* 64均值 : 0.5843214285714285
* 64ens : 0.5892857142857143 
* 120 MSR-3 : 0.5907142857142857 

Color(gap:50)-RF(scale(-1,1))
* 16均值: 0.4855714285714286
* 16ens : 0.5307142857142857
* 32均值 : 0.5233214285714285
* 32ens : 0.5828571428571429
* 64均值 : 0.5510892857142857
* 64ens : 0.6414285714285715
* 120 MSR-3 : 0.6492857142857142
 

Color(gap:50)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.5226964285714284
* 16ens : 0.5807142857142857
* 32均值 : 0.5926607142857143
* 32ens : 0.6164285714285714
* 64均值 : 0.5724464285714286
* 64ens : 0.6221428571428571
* 120 MSR-3 : 0.665
 

#### 3.5.1.3. 三变化

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))
* 16均值: 0.5035357142857142
* 16ens : 0.5564285714285714
* 32均值 : 0.5451964285714286 
* 32ens :0.615
* 64均值 : 0.5567678571428571 
* 64ens : 0.6457142857142857
* 120 MSR-3 : 0.6721428571428572

Color(Gap:50)-RF(scale(-1,1))-GF(sigma: 0-10)
* 16均值:  0.5038571428571429
* 16ens : 0.5521428571428572
* 32均值 : 0.5497321428571429
* 32ens : 0.6071428571428571
* 64均值 :  0.5678392857142858
* 64ens :  0.6492857142857142
* 120 MSR-3 :  0.67


Color(Gap:50)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.5195000000000001
* 16ens : 0.6192857142857143
* 32均值 :0.5806071428571429
* 32ens : 0.645
* 64均值 : 0.558875
* 64ens : 0.6628571428571428
* 120 MSR-3 : 0.7078571428571429


Color(Gap:50)-GN(mean: 0, var 0.01,0.01)-RF(scale(-1,1))
* 16均值: 0.4343571428571429
* 16ens : 0.5992857142857143
* 32均值 : 0.5541607142857142
* 32ens : 0.6457142857142857
* 64均值 : 0.5456071428571428
* 64ens : 0.655
* 120 MSR-3 : 0.7085714285714285

#### 3.5.1.4. 四标准变换 


Color(Gap:50)-GN(mean: 0, var 0.01,0.01)-RF(scale(-1,1))-GF(sigma: 0-10)
* 16均值: 0.5398214285714286 
* 16ens : 0.6457142857142857
* 32均值 : 0.5736964285714286
* 32ens : 0.6571428571428571
* 64均值 : 0.5575178571428572 
* 64ens :  0.6735714285714286
* 120 MSR-3 : 0.7107142857142857


Color(Gap:50)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值:  0.582
* 16ens : 0.6528571428571428
* 32均值 : 0.5973928571428573
* 32ens : 0.6635714285714286
* 64均值 : 0.5707857142857142 
* 64ens : 0.6678571428571428
* 120 MSR-3 : 0.7114285714285714

Color(Gap:50)-RF(scale(-1,1))-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.477625
* 16ens : 0.6035714285714285
* 32均值 : 0.5946964285714286
* 32ens : 0.6614285714285715
* 64均值 : 0.5690714285714287
* 64ens : 0.6707142857142857
* 120 MSR-3 : 0.72s

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)
* 16均值:  0.4923571428571429 
* 16ens : 0.6321428571428571
* 32均值 : 0.5932321428571428
* 32ens : 0.6671428571428571
* 64均值 : 0.5606071428571427
* 64ens : 0.6735714285714286
* 120 MSR-3 : 0.7264285714285714

#### 3.5.1.5. 四标准加入重复的- 单RF

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5893928571428575
* 16ens : 0.6642857142857143
* 32均值 : 0.606107142857143
* 32ens : 0.685
* 64均值 : 0.5713749999999999 
* 64ens : 0.6871428571428572
* 120 MSR-3 : 0.7285714285714285

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.42957142857142855
* 16ens : 0.6035714285714285
* 32均值 : 0.5930892857142859
* 32ens : 0.6764285714285714
* 64均值 : 0.5651428571428572
* 64ens : 0.6878571428571428
* 120 MSR-3 : 0.7364285714285714
 
Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5752857142857144
* 16ens :  0.6721428571428572
* 32均值 : 0.6070178571428572
* 32ens : 0.6871428571428572
* 64均值 : 0.5744285714285713
* 64ens : 0.695 
* 120 MSR-3 : 0.74


#### 3.5.1.6. 四标准加入重复的- 多RF


Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值:  0.44135714285714284 
* 16ens : 0.6535714285714286
* 32均值 :  0.5850892857142858
* 32ens : 0.7114285714285714
* 64均值 :   0.5520535714285716
* 64ens : 0.7071428571428572
* 120 MSR-3 : 0.7557142857142857


Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5358392857142856
* 16ens : 0.6842857142857143
* 32均值 : 0.5775178571428572
* 32ens : 0.7042857142857143
* 64均值 : 0.5383392857142857
* 64ens : 0.6978571428571428
* 120 MSR-3 : 0.7471428571428571


CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.42751785714285706
* 16ens : 0.675
* 32均值 : 0.5695178571428572
* 32ens : 0.72
* 64均值 : 0.5253392857142857
* 64ens : 0.7135714285714285
* 120 MSR-3 : 0.7592857142857142


#### 3.5.1.7. 加入几何

PT(p:1, 0-0,3)-Color(Gap:50)-GF(sigma: 0-10)-RF(-1,1)-GN(mean:0, var: 0.01-0.01):
* 16均值: 0.2902321428571429
* 16ens : 0.5514285714285714
* 32均值 : 0.3886071428571428
* 32ens : 0.6135714285714285
* 64均值 : 0.35160714285714273
* 64ens : 0.5964285714285714
* 120 MSR-3 : 0.6921428571428572

AT(p:1, rotation: 20, shear: x15, y15)-Color(Gap:50)-GF(sigma: 0-10)-RF(-1,1)-GN(mean:0, var: 0.01-0.01):
* 16均值:　0.3181785714285714
* 16ens :　0.5871428571428572
* 32均值 :　0.42912499999999987
* 32ens :　0.6607142857142857
* 64均值 :　0.38626785714285716
* 64ens :　0.6671428571428571
* 120 MSR-3 :　0.7314285714285714


* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :


* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :


### 3.5.2. Deepnetwork


#### 3.5.2.1. 200,000

* 64batch size
* 20epoch

AT(p:1, rotation: 20, shear: x15, y15)-Color(Gap:50)-RF(-1,1)-GN(mean:0, var: 0.01-0.01)-GF(sigma: 0-10):
* MyNet 44 0.5807142857142857 
  * Best : [Epoch: 18] --- Iteration: 54900, Acc: 0.595.
* MyNet 64 收敛失败
* MyNet 84 收敛失败
* ------------------
* Alexnet : 收敛失败
* VGG11 ：收敛失败
* Resnet34 : 0.5814285714285714
  * Best : [Epoch: 20] --- Iteration: 61400, Acc: 0.6121428571428571.




PT(p:1, 0-0,3)-Color(Gap:50)-RF(-1,1)-GN(mean:0, var: 0.01-0.01)-GF(sigma: 0-10)
* MyNet 44 0.5807142857142857 
  * Best : [Epoch: 20] --- Iteration: 61100, Acc: 0.6435714285714286.
* MyNet 64 0.4642857142857143
  * Best : [Epoch: 19] --- Iteration: 56700, Acc: 0.5328571428571428.
* MyNet 84 收敛失败



#### 3.5.2.2. 500,000
* 64batch size
* 20epoch
* Resnet34 : 0.7857142857142857
* MyNet 44 : 0.6685714285714286
  * Best :  [Epoch: 15] --- Iteration: 113200, Acc: 0.6921428571428572.
* MyNet 64 
  * Best : 
* MyNet 84 

PT(p:1, 0-0,3)-Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

# 1. Save Experiment Result


# 2. JPSC1400 Use CNN

## 2.1. 7fontJpan padding=2 比较SSCD生成算法

Base : CC-GF-RF-GF-RF-GF-RF-GF-GN
Base : GF-RF-GF-RF-GF-RF-GF-GN

### 500000 + MyNet44(no33)(DLD) - 40 epoches 

MO(Horie)-PT_dir(-5,5)-CC-GF-RF-GN
Finally top1 accuracy: 0.8121428571428572
Finally top2 accuracy: 0.8757142857142857
Best : [Epoch: 27] --- Iteration: 105300, Acc: 0.835.


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC-GF-RF-GN
Finally top1 accuracy: 0.7914285714285715
Finally top2 accuracy: 0.8671428571428571
Best : [Epoch: 39] --- Iteration: 150000, Acc: 0.8185714285714286.


----------------------------------------------------------
MO(0.7,d4,d8,pU,pD)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8328571428571429
Finally top2 accuracy: 0.9035714285714286
Best : [Epoch: 30] --- Iteration: 116100, Acc: 0.8507142857142858


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8321428571428572
Finally top2 accuracy: 0.8985714285714286
Best : [Epoch: 36] --- Iteration: 139800, Acc: 0.8585714285714285.


MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8314285714285714
Finally top2 accuracy: 0.895
Best : [Epoch: 33] --- Iteration: 127900, Acc: 0.8435714285714285.


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.835
Finally top2 accuracy: 0.9007142857142857
Best : [Epoch: 40] --- Iteration: 154700, Acc: 0.855

----------------------------------------------------------------


### 500000 + MyVGG(no33)(DLD) - 40 epoches 

MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8378571428571429
Finally top2 accuracy: 0.8978571428571429
Best : [Epoch: 37] --- Iteration: 141100, Acc: 0.8528571428571429.




### 2.1.1. 500000 + MyNet44(no33)(DLD)  - LOSS 0.5 (废弃)

MyNet44-512:
MO(0.7)-PT(-0.1,0.2,p=1)-Base: 

Finally correct predicted: 0.8271428571428572
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 62900, Acc: 0.8442857142857143.
[Epoch: 19] --- Iteration: 74233, Loss: 0.47950335680459494.

Finally correct predicted: 0.8521428571428571
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 69000, Acc: 0.8614285714285714.
[Epoch: 20] --- Iteration: 78140, Loss: 0.4976906088335422

Finally correct predicted: 0.8357142857142857
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 70600, Acc: 0.8478571428571429.
[Epoch: 22] --- Iteration: 85954, Loss: 0.4985309371130388.

Finally correct predicted: 0.8278571428571428
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 72200, Acc: 0.8414285714285714.
[Epoch: 19] --- Iteration: 74233, Loss: 0.47821108121250566.

--------------------------------------------

MyNet44-512:
MO(0.7)-PT(-0.1,0.2,p=1)-AT(p=1,shear(15,15),autoscale)-Base: 
Finally correct predicted: 0.8321428571428572
Predict one set cost time: 2 s
Best : [Epoch: 31] --- Iteration: 118500, Acc: 0.8485714285714285.
[Epoch: 35] --- Iteration: 136745, Loss: 0.49174044895602764.

--------------------------------------------

MyNet44-768:
MO(0.7)-PT(-0.1,0.2,p=1)-Base: 

Finally correct predicted: 0.8357142857142857
Predict one set cost time: 2 s
Best : [Epoch: 14] --- Iteration: 54100, Acc: 0.8571428571428571.
[Epoch: 20] --- Iteration: 78140, Loss: 0.48952570915440635.

### 2.1.2. 500000 + MyVGG （废弃）

MO(0.7)-PT(-0.1,0.2,p=1)-Base  
Finally correct predicted: 0.8264285714285714
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38300, Acc: 0.8428571428571429.

MO(0.3)-AF(ro=30,sh=15,p=1,autoscale)-Base


MO(0.3)-AF(ro=20,sh=15,p=1,autoscale)-Base


### 2.1.3. (废弃) 500000 + 50次单bunch  + MyNet 44-512 （with first 33)

Train cost: 2 h 40 m 40 s
Train cost: 2 h 42 m 27 s
Train cost: 2 h 44 m 43 s 
Train cost: 2 h 45 m 10 s 


- MO 概率 0.3----------------------------------------------------------------------
- AF ro=30 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.3)-AF(ro=30,sh=15,p=0.7)-PT(-0.1,0.2,p=1)-Base
Final : 0.7985714285714286
Best  : [Epoch: 21] --- Iteration: 100, Acc: 0.8185714285714286

MO(0.3)-AF(ro=30,sh=15,p=1)-PT(-0.1,0.2,p=1)-Base
Final :0.7978571428571428
Best  : [Epoch: 58] --- Iteration: 100, Acc: 0.7978571428571428

- MO 概率 0.3---------------------------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变
  
MO(0.3)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base
Final : 0.8285714285714286
Best  : [Epoch: 20] --- Iteration: 100, Acc: 0.8457142857142858
MobileNet: 0.8128571428571428
MobileNet Bset : 0.8278571428571428

MO(0.3)-AF(ro=20,sh=15,p=1)-PT(-0.1,0.2,p=1)-Base
Final : 0.8307142857142857
Best  : [Epoch: 36] --- Iteration: 100, Acc: 0.8385714285714285


- MO 概率 0.3 调换AF PT 的顺序  ------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变
MO(0.3)-PT(-0.1,0.2,p=1)-AF(ro=20,sh=15,p=0.5)-Base
Final : 0.8385714285714285
Best  : [Epoch: 28] --- Iteration: 100, Acc: 0.8414285714285714.

MO(0.3)-PT(-0.1,0.2,p=1)-AF(ro=20,sh=15,p=1)-Base
Final : 0.7735714285714286
Best  : [Epoch: 52] --- Iteration: 100, Acc: 0.81


- MO 的概率上升到 0.7 -------------------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.7)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base
Final : 0.8335714285714285
Best  : [Epoch: 47] --- Iteration: 100, Acc: 0.8464285714285714

MO(0.7)-AF(ro=20,sh=15,p=0.7)-PT(-0.1,0.2,p=1)-Base
Final : 0.8128571428571428
Best  : [Epoch: 42] --- Iteration: 100, Acc: 0.8235714285714286


- MO 的概率上升到 0.7 调换AF PT 的顺序  -------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.7)-PT(-0.1,0.2,p=1)--AF(ro=20,sh=15,p=1)Base
Final : 0.8371428571428572
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.835

MO(0.7)-PT(-0.1,0.2,p=1)--AF(ro=20,sh=15,p=0.7)Base
Final : 0.8292857142857143
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8407142857142857



## 2.2. 7fontJpan padding=2 比较网络结构 (训练方法已废弃)

SSCD 都是 MO(0.7,d4,d8,pU)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base

* 标题是训练数据量和方法
* 这里的 MyNet 默认是带 first33 的

### 2.2.1. 500k(adam,0.0001,e=10)

- MyNet42(512)
Finally correct predicted: 0.76
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37000, Acc: 0.7742857142857142.
[Epoch: 10] --- Iteration: 39000, Loss: 3.704668809183133.

- MyNet42(512, no33)
Finally correct predicted: 0.7585714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38700, Acc: 0.77.
[Epoch: 10] --- Iteration: 39000, Loss: 2.662528754743246

------------------------------------------------------------------------

- MyNet44(512)
Finally correct predicted: 0.815
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37200, Acc: 0.8278571428571428
[Epoch: 10] --- Iteration: 39000, Loss: 3.202185431713764

- MyNet44(512, no33)
Finally correct predicted: 0.8135714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 35500, Acc: 0.8342857142857143.
[Epoch: 10] --- Iteration: 39000, Loss: 1.7207405022236113.

- MyNet44(512, no33,LD-L)
Finally correct predicted: 0.8207142857142857
Predict one set cost time: 2 s
Best : [Epoch: 8] --- Iteration: 27400, Acc: 0.8421428571428572.
[Epoch: 10] --- Iteration: 39000, Loss: 0.9498902468694858.

- MyNet44(512, no33,LL-L)
Finally correct predicted: 0.7735714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 36400, Acc: 0.7942857142857143
[Epoch: 10] --- Iteration: 39000, Loss: 1.5581608724384808

- MyNet44(512, no33,LLD-L)
Finally correct predicted: 0.7985714285714286
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 31500, Acc: 0.82.
[Epoch: 10] --- Iteration: 39000, Loss: 1.821282786067605


- MyVGG41(512)
Finally correct predicted: 0.82
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 32800, Acc: 0.83.
[Epoch: 10] --- Iteration: 39000, Loss: 2.2967642378851965

------------------------------------------------------------------------


- MyNet46(512)
Finally correct predicted: 0.7942857142857143
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37600, Acc: 0.8085714285714286.
[Epoch: 10] --- Iteration: 39000, Loss: 3.563677761330914

- MyNet46(512, no33)
Finally correct predicted: 0.8
Predict one set cost time: 2 s
Best : [Epoch: 7] --- Iteration: 27000, Acc: 0.8171428571428572
[Epoch: 10] --- Iteration: 39000, Loss: 1.732441489480197.


- MyNet44-4(512,no33)
Finally correct predicted: 0.7107142857142857
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37400, Acc: 0.72.
[Epoch: 10] --- Iteration: 39000, Loss: 3.538749488861133.

- MyNet44-4(512)
Finally correct predicted: 0.775
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37200, Acc: 0.7835714285714286
[Epoch: 10] --- Iteration: 39000, Loss: 2.160151597893773.


- MyNet46-4(512,no33)
Finally correct predicted: 0.7578571428571429
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 34800, Acc: 0.7757142857142857.
[Epoch: 10] --- Iteration: 39000, Loss: 2.0203742220741816.

- MyNet46-4(512)
Finally correct predicted: 0.7585714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38600, Acc: 0.7714285714285715
[Epoch: 10] --- Iteration: 39000, Loss: 3.484916457338211.



### 2.2.2. 500k(adam,0.0001,e=10) + 10*0.500k(adam,0.0001)

- MyNet44(512)
Final: 0.8157142857142857
Best : [Epoch: 20] --- Iteration: 300, Acc: 0.8364285714285714.

-MyNet44(384)
Final : 0.8285714285714286
Best  : [Epoch: 20] --- Iteration: 200, Acc: 0.8278571428571428


### 2.2.3. 500k(adam,0.0001,e=10) + 10*0.500k(adam,0.001,lineral)
- MyNet42(512)
Finally correct predicted: 0.8057142857142857
Predict one set cost time: 2 s
Best : [Epoch: 52] --- Acc: 0.8128571428571428


### 2.2.4. 500k(adam,0.0001,e=10)-20*0.500k(adam,0.001,lineral)

- MyVGG41(512)
Finally correct predicted: 0.835
Predict one set cost time: 2 s
Best : [Epoch: 28] --- Iteration: 200, Acc: 0.8407142857142857


### 2.2.5. 500k(adam,0.0001,e=10)-50*1bun(adam,0.0001)

- MyNet44(384)
Final : 0.8285714285714286
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8428571428571429

- MyNet44(512)
Final : 0.8335714285714285
Best  : [Epoch: 47] --- Iteration: 100, Acc: 0.8464285714285714

- MyNet46(512)
Final : 0.8278571428571428
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8271428571428572


### 2.2.6. 500k(adam,0.0001,e=10)-50*1bun(adam,0.001,lineral)

- MyVGG41(512)
Finally correct predicted: 0.8321428571428572
Predict one set cost time: 2 s
Best : [Epoch: 55] --- Acc: 0.8421428571428572.

- MyNet42(512)
Finally correct predicted: 0.8057142857142857
Predict one set cost time: 2 s
Best : [Epoch: 52] --- Acc: 0.8128571428571428

## 2.3. 网络结构专门对比

* MO-PT-CC-GF-RF-GF-RF-GF-RF-GF-GN 
* 这里的 MyNet 都不再使用 first 33

### 2.3.1. (废弃) 500k + 10 epoch  发现 10 epoch 还不足够充分收敛 
7font

MyNet42
best:     0.7757142857142857
bestiter: 36000
final:    0.7714285714285715  

MyNet44
best:     0.8278571428571428
bestiter: 25700
final:    0.8157142857142857

MyNet46:
best:     0.8171428571428572
bestiter: 37900
final:    0.8092857142857143

MyNet48:
best:     0.8057142857142857
bestiter: 35500
final:    0.7892857142857143


MobileNetV2:  
Finally correct predicted: 0.7928571428571428
Best : [Epoch: 10] --- Iteration: 38800, Acc: 0.8028571428571428

MyVGG
Finally correct predicted: 0.8264285714285714
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38300, Acc: 0.8428571428571429

ResNet34
Finally correct predicted: 0.8228571428571428
Predict one set cost time: 4 s
Best : [Epoch: 10] --- Iteration: 35800, Acc: 0.8435714285714285

-----------------------------------------------------------------------------------

9font 加入自搜的2个手写字体
MO-PT-CC-GF-RF-GF-RF-GF-RF-GF-GN  
500k + 10 epoch  
| 类型网络 | 2                  | 4                  | 6                  | 8                  | vgg                |
| -------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| best     | 0.8057142857142857 | 0.855              | 0.845              | 0.8292857142857143 | 0.855              |
| bestiter | 37800              | 37300              | 30200              | 38300              | 35700              |
| final    | 0.8007142857142857 | 0.8407142857142857 | 0.8314285714285714 | 0.815              | 0.8307142857142857 |

### 2.3.2. (废弃) 500k + 20 epoch 发现20 epoch 仍然有提升空间

MyNet42(LDLD)
Finally correct predicted: 0.7907142857142857
Predict one set cost time: 2 s
Best : [Epoch: 20] --- Iteration: 78000, Acc: 0.7964285714285714. 

MyNet43(LDLD)
Finally correct predicted: 0.8035714285714286
Predict one set cost time: 2 s
Best : [Epoch: 14] --- Iteration: 52000, Acc: 0.8292857142857143

MyNet43(D)
Finally correct predicted: 0.8164285714285714
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 69900, Acc: 0.8285714285714286

MyNet44(LDLD)
Finally correct predicted: 0.8292857142857143
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 71200, Acc: 0.845.

MyNet44(D)
Finally correct predicted: 0.825
Predict one set cost time: 2 s  
Best : [Epoch: 17] --- Iteration: 65300, Acc: 0.8471428571428572.

MyNet44(DLD)
Finally correct predicted: 0.85
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 62900, Acc: 0.8585714285714285.
Finally correct predicted: 0.8442857142857143
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 65900, Acc: 0.8514285714285714.

MyNet45(LDLD)
Finally correct predicted: 0.8042857142857143
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 64100, Acc: 0.8257142857142857

MyNet45(D)
Finally correct predicted: 0.8435714285714285
Predict one set cost time: 2 s
Best : [Epoch: 16] --- Iteration: 60800, Acc: 0.85

MyNet45(DLD)
Finally correct predicted: 0.84
Predict one set cost time: 2 s
Best : [Epoch: 20] --- Iteration: 77700, Acc: 0.8542857142857143.

MyNet46(DLD)
Finally correct predicted: 0.8292857142857143
Predict one set cost time: 2 s
Best : [Epoch: 16] --- Iteration: 59600, Acc: 0.8528571428571429

MyNet46(LDLD)
Finally correct predicted: 0.8021428571428572
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 66900, Acc: 0.8164285714285714


MyNet47(DLD)
Finally correct predicted: 0.8257142857142857
Predict one set cost time: 2 s
Best : [Epoch: 15] --- Iteration: 55100, Acc: 0.855.

MyNet4(6543)(DLD)
Finally correct predicted: 0.8242857142857143 
Predict one set cost time: 2 s 
2.4. Best : [Epoch: 20] --- Iteration: 75600, Acc: 0.8492857142857143.

-----------------------------------------
MyVGG(LDLD)
Finally correct predicted: 0.8285714285714286
Predict one set cost time: 2 s
Best : [Epoch: 15] --- Iteration: 56300, Acc: 0.8478571428571429.

MyVGG(D)
Finally correct predicted: 0.83
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 71100, Acc: 0.845.

------------------------------------------------

MobileNetV2
Finally correct predicted: 0.815
Predict one set cost time: 3 s
Best : [Epoch: 20] --- Iteration: 76100, Acc: 0.8442857142857143.

ResNet34
Finally correct predicted: 0.8357142857142857
Predict one set cost time: 3 s
Best : [Epoch: 14] --- Iteration: 54200, Acc: 0.8464285714285714

### 2.3.3. 500k + 40 epoch

MyVGG(DLD)
Finally top1 accuracy: 0.8621428571428571
Finally top2 accuracy: 0.9092857142857143
Best : [Epoch: 25] --- Iteration: 96700, Acc: 0.8742857142857143.

MyNet44(DLD)
Finally top1 accuracy: 0.8478571428571429
Finally top2 accuracy: 0.9042857142857142
Best : [Epoch: 31] --- Iteration: 120400, Acc: 0.86. 

MyNet43(DLD)
Finally top1 accuracy: 0.8464285714285714
Finally top2 accuracy: 0.9057142857142857
Best : [Epoch: 35] --- Iteration: 136200, Acc: 0.8571428571428571.

MyNet45(DLD)
Finally top1 accuracy: 0.8521428571428571
Finally top2 accuracy: 0.9042857142857142
Best : [Epoch: 39] --- Iteration: 151900, Acc: 0.8592857142857143

MyNet46(DLD)
Finally top1 accuracy: 0.861
Best: 0.869



### 2.3.4. 500K + 相同训练loss

# 3. JPSC1400 Use NNS 120

* 噪点非常重要
* HOG对模糊不敏感
* 随机滤波之后再加噪点效果更好
* 16对噪点非常敏感, GF放在最后对16的精度有提升, 但是对全局精度没好处

* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   

## 3.1. 7fontJpan padding=0

### 3.1.1. NNS

16 : 0.5121428571428571
32 : 0.5414285714285715
64 : 0.5342857142857143

## 3.2. 7fontJpan padding=1
16 : 0.5192857142857142
32 : 0.5371428571428571
64 : 0.5585714285714286
### 3.2.1. NNS

## 3.3. 7fontJpan padding=2


16 : 0.5078571428571429
32 : 0.5257142857142857
64 : 0.5692857142857143

### 3.3.1. 和 Horie 手法比较 种子归一后的数据

CC-GF-RF-GN (种子设定一致)
* 16均值:       0.4899285714285713
* 16ens :       0.6264285714285714
* 32均值 :      0.5903035714285715
* 32ens :       0.665
* 64均值 :      0.561625
* 64ens :       0.6742857142857143
* 120 MSR-3 :   0.7171428571428572

CC(lib20)-GF-RF-GN  (种子设定一致)
* 16均值:       0.5327321428571429 
* 16ens :       0.6507142857142857
* 32均值 :      0.5862499999999998
* 32ens :       0.66
* 64均值 :      0.5587500000000001
* 64ens :       0.6728571428571428
* 120 MSR-3 :   0.7278571428571429

-----------------------------

MO(horie)-CC-GF-RF-GN                   (种子设定一致)
* 16均值:       0.4791964285714285
* 16ens :       0.6521428571428571
* 32均值 :      0.5881607142857144
* 32ens :       0.6985714285714286
* 64均值 :      0.5449285714285714
* 64ens :       0.675
* 120 MSR-3 :   0.7342857142857143

<!-- 
MO(0.7,d4,d8,pU,pD)-CC-GF-RF-GN         (种子设定一致)
* 16均值:       0.4729107142857144
* 16ens :       0.6435714285714286
* 32均值 :      0.5964107142857145
* 32ens :       0.6892857142857143
* 64均值 :      0.5581964285714286
* 64ens :       0.6735714285714286
* 120 MSR-3 :   0.7407142857142858 -->

MO(horie)-CC(lib)-GF-RF-GN              (种子设定一致)
* 16均值:       0.5213035714285715
* 16ens :       0.6742857142857143
* 32均值 :      0.5852678571428569
* 32ens :       0.6914285714285714
* 64均值 :      0.5481964285714286
* 64ens :       0.6842857142857143
* 120 MSR-3 :   0.7414285714285714


MO(0.7,d4,d8,pU,pD)-CC(lib)-GF-RF-GN   (种子设定一致)
* 16均值:       0.5193392857142858
* 16ens :       0.66
* 32均值 :      0.5911607142857144
* 32ens :       0.6878571428571428
* 64均值 :      0.5555892857142857
* 64ens :       0.6764285714285714
* 120 MSR-3 :   0.7364285714285714


MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN   (种子设定一致)
* 16均值:       0.47167857142857156
* 16ens :       0.6471428571428571
* 32均值 :      0.5979107142857143
* 32ens :       0.6928571428571428
* 64均值 :      0.5573392857142857
* 64ens :       0.6678571428571428
* 120 MSR-3 :   0.7342857142857143

----------------------------------------------------------------

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GN   (种子设定一致)
* 16均值:       0.44885714285714295
* 16ens :       0.6614285714285715
* 32均值 :      0.5804107142857142
* 32ens :       0.7085714285714285
* 64均值 :      0.5351964285714284
* 64ens :       0.6878571428571428
* 120 MSR-3 :   0.7471428571428571

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.42716071428571434
* 16ens :       0.6471428571428571
* 32均值 :      0.5867321428571429
* 32ens :       0.7157142857142857
* 64均值 :      0.5428392857142856
* 64ens :       0.7
* 120 MSR-3 :   0.7528571428571429

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GN   (种子设定一致)
* 16均值:       0.4278928571428571
* 16ens :       0.6628571428571428
* 32均值 :      0.5610535714285715
* 32ens :       0.7128571428571429
* 64均值 :      0.5139285714285712
* 64ens :       0.7028571428571428
* 120 MSR-3 :   0.7542857142857143

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN      (种子设定一致)
* 16均值:       0.41089285714285706
* 16ens :       0.6557142857142857
* 32均值 :      0.5640892857142858
* 32ens :       0.7207142857142858
* 64均值 :      0.5198392857142857
* 64ens :       0.7092857142857143
* 120 MSR-3 :   0.7685714285714286

MO(0.7,d4,d8,pU,pD)-PT(dir,-5,5)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.34992857142857153
* 16ens :       0.6514285714285715
* 32均值 :      0.4982678571428571
* 32ens :       0.7414285714285714
* 64均值 :      0.45125000000000004
* 64ens :       0.7257142857142858
* 120 MSR-3 :   0.7807142857142857

MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.3835535714285713
* 16ens :       0.69
* 32均值 :      0.48410714285714285
* 32ens :       0.735
* 64均值 :      0.43892857142857145
* 64ens :       0.7292857142857143
* 120 MSR-3 :   0.7935714285714286


* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   

* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   


--------------------------------------------------------------------------

PT(-0.1,0.2)-CC-GF-RF-GN
* 16均值:     0.39180357142857136
* 16ens :     0.6435714285714286
* 32均值 :    0.5076785714285714
* 32ens :     0.6971428571428572
* 64均值 :    0.47101785714285727
* 64ens :     0.7114285714285714
* 120 MSR-3 : 0.7621428571428571

PT_dir(-5,5)-CC-GF-RF-GN
* 16均值:     0.4165
* 16ens :     0.6564285714285715
* 32均值 :    0.5364464285714287
* 32ens :     0.7185714285714285
* 64均值 :    0.48914285714285705
* 64ens :     0.725
* 120 MSR-3 : 0.7635714285714286



------------------------------------------------
PT_dir(-5,5)-MO(horie)-CC(lib20)-GF-RF-GN
* 16均值:     0.4498214285714286
* 16ens :     0.7107142857142857
* 32均值 :     0.5159464285714284
* 32ens :     0.7292857142857143
* 64均值 :    0.47285714285714275
* 64ens :     0.7214285714285714
* 120 MSR-3 : 0.7771428571428571


PT(-0.1,0.2)-MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN
* 16均值:     0.32730357142857147
* 16ens :     0.6707142857142857
* 32均值 :    0.4996607142857143
* 32ens :     0.725
* 64均值 :    0.4710357142857142
* 64ens :     0.7278571428571429
* 120 MSR-3 : 0.7828571428571428  

Reversed: 
MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.376107142857143
* 16ens :     0.6535714285714286
* 32均值 :    0.5099464285714286
* 32ens :     0.705
* 64均值 :    0.47401785714285716
* 64ens :     0.7285714285714285
* 120 MSR-3 : 0.7664285714285715
------------------------------------------------------


MO(0.7,d4,d8,pU,pD)-PT_dir(-5,5)-CC(gap:50)-GF-RF-GN
* 16均值:     0.3997500000000001
* 16ens :     0.6735714285714286
* 32均值 :    0.5379821428571427
* 32ens :     0.7242857142857143
* 64均值 :    0.4885357142857143
* 64ens :     0.7228571428571429
* 120 MSR-3 : 0.7735714285714286

MO(horie)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.3756785714285714
* 16ens :     0.66
* 32均值 :    0.49557142857142866
* 32ens :     0.7214285714285714
* 64均值 :    0.4543750000000001
* 64ens :     0.7242857142857143
* 120 MSR-3 : 0.7721428571428571

MO(horie)-PT_dir(-5,5)-CC(gap:50)-GF-RF-GN
* 16均值:     0.40514285714285714
* 16ens :     0.6885714285714286
* 32均值 :    0.5240714285714286
* 32ens :     0.7278571428571429
* 64均值 :    0.47648214285714274
* 64ens :     0.73
* 120 MSR-3 : 0.7778571428571428

### 3.3.2. 少变换

CC
* 16均值: 0.5082321428571429
* 16ens :  0.5085714285714286
* 32均值 : 0.5293928571428571
* 32ens :  0.5307142857142857
* 64均值 : 0.5705178571428572
* 64ens : 0.5757142857142857
* 120 MSR-3 : 0.5821428571428572

CC-GN
* 16均值: 0.5229821428571428
* 16ens : 0.5835714285714285
* 32均值 : 0.5901071428571428
* 32ens : 0.6121428571428571
* 64均值 : 0.5714821428571428
* 64ens : 0.6142857142857143
* 120 MSR-3 : 0.6607142857142857

CC-RF-GN
* 16均值: 0.5163749999999998
* 16ens : 0.6107142857142858
* 32均值 : 0.5778392857142857
* 32ens : 0.6442857142857142
* 64均值 : 0.5580178571428569
* 64ens : 0.6564285714285715
* 120 MSR-3 : 0.705

### 3.3.3. 四标准变化单RF

CC-GF-RF-GN
* 16均值: 0.49169642857142853
* 16ens : 0.6307142857142857
* 32均值 : 0.5879107142857142
* 32ens : 0.66
* 64均值 : 0.5593749999999998
* 64ens : 0.67
* 120 MSR-3 : 0.7235714285714285
* 120 MSR-3 : 0.7228571428571429 (第二次)

CC(lib20)-GF-RF-GN
* 16均值: 0.5299285714285713
* 16ens : 0.6464285714285715
* 32均值 : 0.5845178571428572
* 32ens : 0.6685714285714286
* 64均值 : 0.557392857142857
* 64ens : 0.6728571428571428
* 120 MSR-3 : 0.7185714285714285

CC-GF-RF-GN-GF
* 16均值: 0.5853750000000001
* 16ens : 0.6528571428571428
* 32均值 : 0.6016250000000001
* 32ens : 0.6664285714285715
* 64均值 : 0.5709107142857145
* 64ens : 0.6764285714285714
* 120 MSR-3 : 0.7214285714285714  反而还降了，

CC-GF-RF-GF-GN
* 16均值: 0.4588214285714285
* 16ens : 0.6235714285714286
* 32均值 : 0.5991428571428573
* 32ens : 0.6778571428571428
* 64均值 : 0.5685535714285714
* 64ens : 0.685
* 120 MSR-3 : 0.7321428571428571

CC-GF-RF-GN-GF-GN-GF
* 16均值:  0.5751071428571428
* 16ens : 0.6664285714285715
* 32均值 : 0.6082499999999998
* 32ens : 0.6864285714285714
* 64均值 : 0.5711964285714288
* 64ens : 0.685
* 120 MSR-3 : 0.7271428571428571


### 3.3.4. 四标准多RF

CC-GF-RF-GF-RF-GF-RF-GF-GN

CC-GF-RF-GF-RF-GF-GN
* 16均值: 0.4432142857142859
* 16ens : 0.6521428571428571
* 32均值 : 0.5862142857142858
* 32ens : 0.7157142857142857
* 64均值 : 0.5513571428571428
* 64ens : 0.7028571428571428
* 120 MSR-3 :  0.7535714285714286

CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.4226964285714286
* 16ens : 0.655
* 32均值 : 0.56625
* 32ens : 0.7192857142857143
* 64均值 : 0.5238571428571428
* 64ens :  0.7078571428571429 
* 120 MSR-3 : 0.7578571428571429

CC-GF-RF-GF-RF-GF-RF-GF-GN (2)
* 16均值: 0.42312500000000003
* 16ens : 0.6628571428571428
* 32均值 : 0.5685714285714285
* 32ens : 0.7207142857142858
* 64均值 : 0.528357142857143
* 64ens : 0.71
* 120 MSR-3 : 0.7621428571428571



### 3.3.5. 加入单几何或形态学

-----------------------MO-------------------------

MO(p=0.3, d4,d8,pU)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.42207142857142854
* 16ens : 0.6714285714285714
* 32均值 : 0.5697321428571427
* 32ens : 0.7271428571428571
* 64均值 : 0.523857142857143
* 64ens : 0.7085714285714285
* 120 MSR-3 : 0.7685714285714286

MO(p=0.3, d4,d8,pU)-CC-GF-RF-GF-RF-GF-RF-GF-GN (2)
* 16均值: 0.42207142857142854
* 16ens : 0.6714285714285714
* 32均值 : 0.5697321428571427
* 32ens : 0.7271428571428571
* 64均值 : 0.523857142857143
* 64ens : 0.7085714285714285
* 120 MSR-3 : 0.7685714285714286

---------------------------AF --------------------------------

AF(p=1, ro=20, sh=(10,10))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.28858928571428577
* 16ens : 0.6257142857142857
* 32均值 : 0.4144821428571429
* 32ens : 0.6992857142857143
* 64均值 : 0.373482142857143
* 64ens : 0.685
* 120 MSR-3 : 0.755

 
AF(p=1, ro=20, sh=(15,15))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.27005357142857145
* 16ens : 0.6264285714285714
* 32均值 : 0.3912321428571428
* 32ens : 0.695
* 64均值 : 0.35505357142857147
* 64ens : 0.6971428571428572
* 120 MSR-3 : 0.7592857142857142

AF(p=1, ro=30, sh=(10,10))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.24335714285714288
* 16ens : 0.5835714285714285
* 32均值 : 0.36321428571428577
* 32ens : 0.6907142857142857
* 64均值 : 0.31519642857142854
* 64ens :  0.67
* 120 MSR-3 : 0.7464285714285714

AF(p=1, ro=30, sh=(15,15))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.22530357142857146
* 16ens : 0.5657142857142857
* 32均值 : 0.33825
* 32ens :  0.6578571428571428
* 64均值 : 0.30189285714285713
* 64ens :  0.6621428571428571
* 120 MSR-3 : 0.7321428571428571

-----------------------------------------PT--------------------------

PT(0,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.38489285714285715
* 16ens : 0.665
* 32均值 : 0.5291428571428571
* 32ens : 0.7135714285714285
* 64均值 : 0.5013928571428572
* 64ens :  0.7114285714285714
* 120 MSR-3 : 0.7685714285714286

PT(0,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3215892857142858
* 16ens : 0.615 
* 32均值 : 0.45326785714285717
* 32ens : 0.6921428571428572
* 64均值 : 0.42474999999999985
* 64ens : 0.6764285714285714
* 120 MSR-3 : 0.745

PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3964285714285714
* 16ens : 0.6664285714285715
* 32均值 : 0.5350535714285714
* 32ens : 0.7228571428571429
* 64均值 : 0.4935892857142858
* 64ens : 0.7278571428571429
* 120 MSR-3 : 0.7742857142857142

PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3322142857142857
* 16ens : 0.6571428571428571
* 32均值 : 0.47275
* 32ens : 0.7228571428571429
* 64均值 : 0.434875
* 64ens :  0.7128571428571429
* 120 MSR-3 : 0.77
 
### 3.3.6. 形态学+投影变换


MO(p=0.3, d4,d8,pU)-PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :

MO(p=0.7, d4,d8,pU)-PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.38658928571428575
* 16ens : 0.6757142857142857
* 32均值 : 0.5393035714285717
* 32ens : 0.7314285714285714
* 64均值 : 0.48953571428571435
* 64ens :  0.7157142857142857
* 120 MSR-3 : 0.7792857142857142


MO(p=0.3, d4,d8,pU)-PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.32866071428571425
* 16ens : 0.6678571428571428
* 32均值 : 0.4697321428571429
* 32ens : 0.7307142857142858
* 64均值 : 0.4373214285714284
* 64ens :  0.7292857142857143
* 120 MSR-3 : 0.7821428571428571

MO(p=0.7, d4,d8,pU)-PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.3263214285714286
* 16ens :  0.6578571428571428
* 32均值 : 0.4781607142857144
* 32ens :  0.735
* 64均值 : 0.43808928571428574
* 64ens :  0.7171428571428572
* 120 MSR-3 : 0.7821428571428571

--------------------------------------------

PT(-0.1,0.2)-MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN
* 16均值:     0.32730357142857147
* 16ens :     0.6707142857142857
* 32均值 :    0.4996607142857143
* 32ens :     0.725
* 64均值 :    0.4710357142857142
* 64ens :     0.7278571428571429
* 120 MSR-3 : 0.7828571428571428  

Reversed: 
MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.376107142857143
* 16ens :     0.6535714285714286
* 32均值 :    0.5099464285714286
* 32ens :     0.705
* 64均值 :    0.47401785714285716
* 64ens :     0.7285714285714285
* 120 MSR-3 : 0.7664285714285715



## 3.4. 7fontJpan revised  3097

## 3.5. 7fontJpan padding=2 revised  3104　ヘぺべ　已舍弃 

### 3.5.1. NNS

* 16 ：0.51
* 32 ：0.5278571428571428
* 64 ：0.5707142857142857

* BaseLine : 57



#### 3.5.1.1. 单变换 baseline

颜色 gap 50:
* 16均值:   0.5107857142857143
* 16ens :  0.5107142857142857
* 32均值 : 0.5310535714285713
* 32ens :  0.5328571428571428
* 64均值 :  0.5730357142857144
* 64ens :  0.575
* 120 MSR-3 : 0.5707142857142857

#### 3.5.1.2. 双变换 

Color(gap:50)-GF(sigma: 0-10)
* 16均值: 0.5099107142857142
* 16ens : 0.515
* 32均值 : 0.5586785714285714 
* 32ens : 0.5607142857142857
* 64均值 : 0.5843214285714285
* 64ens : 0.5892857142857143 
* 120 MSR-3 : 0.5907142857142857 

Color(gap:50)-RF(scale(-1,1))
* 16均值: 0.4855714285714286
* 16ens : 0.5307142857142857
* 32均值 : 0.5233214285714285
* 32ens : 0.5828571428571429
* 64均值 : 0.5510892857142857
* 64ens : 0.6414285714285715
* 120 MSR-3 : 0.6492857142857142
 

Color(gap:50)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.5226964285714284
* 16ens : 0.5807142857142857
* 32均值 : 0.5926607142857143
* 32ens : 0.6164285714285714
* 64均值 : 0.5724464285714286
* 64ens : 0.6221428571428571
* 120 MSR-3 : 0.665
 

#### 3.5.1.3. 三变化

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))
* 16均值: 0.5035357142857142
* 16ens : 0.5564285714285714
* 32均值 : 0.5451964285714286 
* 32ens :0.615
* 64均值 : 0.5567678571428571 
* 64ens : 0.6457142857142857
* 120 MSR-3 : 0.6721428571428572

Color(Gap:50)-RF(scale(-1,1))-GF(sigma: 0-10)
* 16均值:  0.5038571428571429
* 16ens : 0.5521428571428572
* 32均值 : 0.5497321428571429
* 32ens : 0.6071428571428571
* 64均值 :  0.5678392857142858
* 64ens :  0.6492857142857142
* 120 MSR-3 :  0.67


Color(Gap:50)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.5195000000000001
* 16ens : 0.6192857142857143
* 32均值 :0.5806071428571429
* 32ens : 0.645
* 64均值 : 0.558875
* 64ens : 0.6628571428571428
* 120 MSR-3 : 0.7078571428571429


Color(Gap:50)-GN(mean: 0, var 0.01,0.01)-RF(scale(-1,1))
* 16均值: 0.4343571428571429
* 16ens : 0.5992857142857143
* 32均值 : 0.5541607142857142
* 32ens : 0.6457142857142857
* 64均值 : 0.5456071428571428
* 64ens : 0.655
* 120 MSR-3 : 0.7085714285714285

#### 3.5.1.4. 四标准变换 


Color(Gap:50)-GN(mean: 0, var 0.01,0.01)-RF(scale(-1,1))-GF(sigma: 0-10)
* 16均值: 0.5398214285714286 
* 16ens : 0.6457142857142857
* 32均值 : 0.5736964285714286
* 32ens : 0.6571428571428571
* 64均值 : 0.5575178571428572 
* 64ens :  0.6735714285714286
* 120 MSR-3 : 0.7107142857142857


Color(Gap:50)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值:  0.582
* 16ens : 0.6528571428571428
* 32均值 : 0.5973928571428573
* 32ens : 0.6635714285714286
* 64均值 : 0.5707857142857142 
* 64ens : 0.6678571428571428
* 120 MSR-3 : 0.7114285714285714

Color(Gap:50)-RF(scale(-1,1))-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.477625
* 16ens : 0.6035714285714285
* 32均值 : 0.5946964285714286
* 32ens : 0.6614285714285715
* 64均值 : 0.5690714285714287
* 64ens : 0.6707142857142857
* 120 MSR-3 : 0.72s

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)
* 16均值:  0.4923571428571429 
* 16ens : 0.6321428571428571
* 32均值 : 0.5932321428571428
* 32ens : 0.6671428571428571
* 64均值 : 0.5606071428571427
* 64ens : 0.6735714285714286
* 120 MSR-3 : 0.7264285714285714

#### 3.5.1.5. 四标准加入重复的- 单RF

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5893928571428575
* 16ens : 0.6642857142857143
* 32均值 : 0.606107142857143
* 32ens : 0.685
* 64均值 : 0.5713749999999999 
* 64ens : 0.6871428571428572
* 120 MSR-3 : 0.7285714285714285

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.42957142857142855
* 16ens : 0.6035714285714285
* 32均值 : 0.5930892857142859
* 32ens : 0.6764285714285714
* 64均值 : 0.5651428571428572
* 64ens : 0.6878571428571428
* 120 MSR-3 : 0.7364285714285714
 
Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5752857142857144
* 16ens :  0.6721428571428572
* 32均值 : 0.6070178571428572
* 32ens : 0.6871428571428572
* 64均值 : 0.5744285714285713
* 64ens : 0.695 
* 120 MSR-3 : 0.74


#### 3.5.1.6. 四标准加入重复的- 多RF


Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值:  0.44135714285714284 
* 16ens : 0.6535714285714286
* 32均值 :  0.5850892857142858
* 32ens : 0.7114285714285714
* 64均值 :   0.5520535714285716
* 64ens : 0.7071428571428572
* 120 MSR-3 : 0.7557142857142857


Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5358392857142856
* 16ens : 0.6842857142857143
* 32均值 : 0.5775178571428572
* 32ens : 0.7042857142857143
* 64均值 : 0.5383392857142857
* 64ens : 0.6978571428571428
* 120 MSR-3 : 0.7471428571428571


CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.42751785714285706
* 16ens : 0.675
* 32均值 : 0.5695178571428572
* 32ens : 0.72
* 64均值 : 0.5253392857142857
* 64ens : 0.7135714285714285
* 120 MSR-3 : 0.7592857142857142


#### 3.5.1.7. 加入几何

PT(p:1, 0-0,3)-Color(Gap:50)-GF(sigma: 0-10)-RF(-1,1)-GN(mean:0, var: 0.01-0.01):
* 16均值: 0.2902321428571429
* 16ens : 0.5514285714285714
* 32均值 : 0.3886071428571428
* 32ens : 0.6135714285714285
* 64均值 : 0.35160714285714273
* 64ens : 0.5964285714285714
* 120 MSR-3 : 0.6921428571428572

AT(p:1, rotation: 20, shear: x15, y15)-Color(Gap:50)-GF(sigma: 0-10)-RF(-1,1)-GN(mean:0, var: 0.01-0.01):
* 16均值:　0.3181785714285714
* 16ens :　0.5871428571428572
* 32均值 :　0.42912499999999987
* 32ens :　0.6607142857142857
* 64均值 :　0.38626785714285716
* 64ens :　0.6671428571428571
* 120 MSR-3 :　0.7314285714285714


* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :


* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :


### 3.5.2. Deepnetwork


#### 3.5.2.1. 200,000

* 64batch size
* 20epoch

AT(p:1, rotation: 20, shear: x15, y15)-Color(Gap:50)-RF(-1,1)-GN(mean:0, var: 0.01-0.01)-GF(sigma: 0-10):
* MyNet 44 0.5807142857142857 
  * Best : [Epoch: 18] --- Iteration: 54900, Acc: 0.595.
* MyNet 64 收敛失败
* MyNet 84 收敛失败
* ------------------
* Alexnet : 收敛失败
* VGG11 ：收敛失败
* Resnet34 : 0.5814285714285714
  * Best : [Epoch: 20] --- Iteration: 61400, Acc: 0.6121428571428571.




PT(p:1, 0-0,3)-Color(Gap:50)-RF(-1,1)-GN(mean:0, var: 0.01-0.01)-GF(sigma: 0-10)
* MyNet 44 0.5807142857142857 
  * Best : [Epoch: 20] --- Iteration: 61100, Acc: 0.6435714285714286.
* MyNet 64 0.4642857142857143
  * Best : [Epoch: 19] --- Iteration: 56700, Acc: 0.5328571428571428.
* MyNet 84 收敛失败



#### 3.5.2.2. 500,000
* 64batch size
* 20epoch
* Resnet34 : 0.7857142857142857
* MyNet 44 : 0.6685714285714286
  * Best :  [Epoch: 15] --- Iteration: 113200, Acc: 0.6921428571428572.
* MyNet 64 
  * Best : 
* MyNet 84 

PT(p:1, 0-0,3)-Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

# 1. Save Experiment Result


# 2. JPSC1400 Use CNN

## 2.1. 7fontJpan padding=2 比较SSCD生成算法

Base : CC-GF-RF-GF-RF-GF-RF-GF-GN
Base : GF-RF-GF-RF-GF-RF-GF-GN

### 500000 + MyNet44(no33)(DLD) - 40 epoches 

MO(Horie)-PT_dir(-5,5)-CC-GF-RF-GN
Finally top1 accuracy: 0.8121428571428572
Finally top2 accuracy: 0.8757142857142857
Best : [Epoch: 27] --- Iteration: 105300, Acc: 0.835.


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC-GF-RF-GN
Finally top1 accuracy: 0.7914285714285715
Finally top2 accuracy: 0.8671428571428571
Best : [Epoch: 39] --- Iteration: 150000, Acc: 0.8185714285714286.


----------------------------------------------------------
MO(0.7,d4,d8,pU,pD)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8328571428571429
Finally top2 accuracy: 0.9035714285714286
Best : [Epoch: 30] --- Iteration: 116100, Acc: 0.8507142857142858


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8321428571428572
Finally top2 accuracy: 0.8985714285714286
Best : [Epoch: 36] --- Iteration: 139800, Acc: 0.8585714285714285.


MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8314285714285714
Finally top2 accuracy: 0.895
Best : [Epoch: 33] --- Iteration: 127900, Acc: 0.8435714285714285.


MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.835
Finally top2 accuracy: 0.9007142857142857
Best : [Epoch: 40] --- Iteration: 154700, Acc: 0.855

----------------------------------------------------------------


### 500000 + MyVGG(no33)(DLD) - 40 epoches 

MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN
Finally top1 accuracy: 0.8378571428571429
Finally top2 accuracy: 0.8978571428571429
Best : [Epoch: 37] --- Iteration: 141100, Acc: 0.8528571428571429.




### 2.1.1. 500000 + MyNet44(no33)(DLD)  - LOSS 0.5 (废弃)

MyNet44-512:
MO(0.7)-PT(-0.1,0.2,p=1)-Base: 

Finally correct predicted: 0.8271428571428572
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 62900, Acc: 0.8442857142857143.
[Epoch: 19] --- Iteration: 74233, Loss: 0.47950335680459494.

Finally correct predicted: 0.8521428571428571
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 69000, Acc: 0.8614285714285714.
[Epoch: 20] --- Iteration: 78140, Loss: 0.4976906088335422

Finally correct predicted: 0.8357142857142857
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 70600, Acc: 0.8478571428571429.
[Epoch: 22] --- Iteration: 85954, Loss: 0.4985309371130388.

Finally correct predicted: 0.8278571428571428
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 72200, Acc: 0.8414285714285714.
[Epoch: 19] --- Iteration: 74233, Loss: 0.47821108121250566.

--------------------------------------------

MyNet44-512:
MO(0.7)-PT(-0.1,0.2,p=1)-AT(p=1,shear(15,15),autoscale)-Base: 
Finally correct predicted: 0.8321428571428572
Predict one set cost time: 2 s
Best : [Epoch: 31] --- Iteration: 118500, Acc: 0.8485714285714285.
[Epoch: 35] --- Iteration: 136745, Loss: 0.49174044895602764.

--------------------------------------------

MyNet44-768:
MO(0.7)-PT(-0.1,0.2,p=1)-Base: 

Finally correct predicted: 0.8357142857142857
Predict one set cost time: 2 s
Best : [Epoch: 14] --- Iteration: 54100, Acc: 0.8571428571428571.
[Epoch: 20] --- Iteration: 78140, Loss: 0.48952570915440635.

### 2.1.2. 500000 + MyVGG （废弃）

MO(0.7)-PT(-0.1,0.2,p=1)-Base  
Finally correct predicted: 0.8264285714285714
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38300, Acc: 0.8428571428571429.

MO(0.3)-AF(ro=30,sh=15,p=1,autoscale)-Base


MO(0.3)-AF(ro=20,sh=15,p=1,autoscale)-Base


### 2.1.3. (废弃) 500000 + 50次单bunch  + MyNet 44-512 （with first 33)

Train cost: 2 h 40 m 40 s
Train cost: 2 h 42 m 27 s
Train cost: 2 h 44 m 43 s 
Train cost: 2 h 45 m 10 s 


- MO 概率 0.3----------------------------------------------------------------------
- AF ro=30 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.3)-AF(ro=30,sh=15,p=0.7)-PT(-0.1,0.2,p=1)-Base
Final : 0.7985714285714286
Best  : [Epoch: 21] --- Iteration: 100, Acc: 0.8185714285714286

MO(0.3)-AF(ro=30,sh=15,p=1)-PT(-0.1,0.2,p=1)-Base
Final :0.7978571428571428
Best  : [Epoch: 58] --- Iteration: 100, Acc: 0.7978571428571428

- MO 概率 0.3---------------------------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变
  
MO(0.3)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base
Final : 0.8285714285714286
Best  : [Epoch: 20] --- Iteration: 100, Acc: 0.8457142857142858
MobileNet: 0.8128571428571428
MobileNet Bset : 0.8278571428571428

MO(0.3)-AF(ro=20,sh=15,p=1)-PT(-0.1,0.2,p=1)-Base
Final : 0.8307142857142857
Best  : [Epoch: 36] --- Iteration: 100, Acc: 0.8385714285714285


- MO 概率 0.3 调换AF PT 的顺序  ------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变
MO(0.3)-PT(-0.1,0.2,p=1)-AF(ro=20,sh=15,p=0.5)-Base
Final : 0.8385714285714285
Best  : [Epoch: 28] --- Iteration: 100, Acc: 0.8414285714285714.

MO(0.3)-PT(-0.1,0.2,p=1)-AF(ro=20,sh=15,p=1)-Base
Final : 0.7735714285714286
Best  : [Epoch: 52] --- Iteration: 100, Acc: 0.81


- MO 的概率上升到 0.7 -------------------------------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.7)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base
Final : 0.8335714285714285
Best  : [Epoch: 47] --- Iteration: 100, Acc: 0.8464285714285714

MO(0.7)-AF(ro=20,sh=15,p=0.7)-PT(-0.1,0.2,p=1)-Base
Final : 0.8128571428571428
Best  : [Epoch: 42] --- Iteration: 100, Acc: 0.8235714285714286


- MO 的概率上升到 0.7 调换AF PT 的顺序  -------------------------------------
- AF ro=20 sh=15 调整概率 , PT (-0.1,0.2,p=1) 不变

MO(0.7)-PT(-0.1,0.2,p=1)--AF(ro=20,sh=15,p=1)Base
Final : 0.8371428571428572
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.835

MO(0.7)-PT(-0.1,0.2,p=1)--AF(ro=20,sh=15,p=0.7)Base
Final : 0.8292857142857143
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8407142857142857



## 2.2. 7fontJpan padding=2 比较网络结构 (训练方法已废弃)

SSCD 都是 MO(0.7,d4,d8,pU)-AF(ro=20,sh=15,p=0.5)-PT(-0.1,0.2,p=1)-Base

* 标题是训练数据量和方法
* 这里的 MyNet 默认是带 first33 的

### 2.2.1. 500k(adam,0.0001,e=10)

- MyNet42(512)
Finally correct predicted: 0.76
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37000, Acc: 0.7742857142857142.
[Epoch: 10] --- Iteration: 39000, Loss: 3.704668809183133.

- MyNet42(512, no33)
Finally correct predicted: 0.7585714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38700, Acc: 0.77.
[Epoch: 10] --- Iteration: 39000, Loss: 2.662528754743246

------------------------------------------------------------------------

- MyNet44(512)
Finally correct predicted: 0.815
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37200, Acc: 0.8278571428571428
[Epoch: 10] --- Iteration: 39000, Loss: 3.202185431713764

- MyNet44(512, no33)
Finally correct predicted: 0.8135714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 35500, Acc: 0.8342857142857143.
[Epoch: 10] --- Iteration: 39000, Loss: 1.7207405022236113.

- MyNet44(512, no33,LD-L)
Finally correct predicted: 0.8207142857142857
Predict one set cost time: 2 s
Best : [Epoch: 8] --- Iteration: 27400, Acc: 0.8421428571428572.
[Epoch: 10] --- Iteration: 39000, Loss: 0.9498902468694858.

- MyNet44(512, no33,LL-L)
Finally correct predicted: 0.7735714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 36400, Acc: 0.7942857142857143
[Epoch: 10] --- Iteration: 39000, Loss: 1.5581608724384808

- MyNet44(512, no33,LLD-L)
Finally correct predicted: 0.7985714285714286
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 31500, Acc: 0.82.
[Epoch: 10] --- Iteration: 39000, Loss: 1.821282786067605


- MyVGG41(512)
Finally correct predicted: 0.82
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 32800, Acc: 0.83.
[Epoch: 10] --- Iteration: 39000, Loss: 2.2967642378851965

------------------------------------------------------------------------


- MyNet46(512)
Finally correct predicted: 0.7942857142857143
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37600, Acc: 0.8085714285714286.
[Epoch: 10] --- Iteration: 39000, Loss: 3.563677761330914

- MyNet46(512, no33)
Finally correct predicted: 0.8
Predict one set cost time: 2 s
Best : [Epoch: 7] --- Iteration: 27000, Acc: 0.8171428571428572
[Epoch: 10] --- Iteration: 39000, Loss: 1.732441489480197.


- MyNet44-4(512,no33)
Finally correct predicted: 0.7107142857142857
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37400, Acc: 0.72.
[Epoch: 10] --- Iteration: 39000, Loss: 3.538749488861133.

- MyNet44-4(512)
Finally correct predicted: 0.775
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 37200, Acc: 0.7835714285714286
[Epoch: 10] --- Iteration: 39000, Loss: 2.160151597893773.


- MyNet46-4(512,no33)
Finally correct predicted: 0.7578571428571429
Predict one set cost time: 2 s
Best : [Epoch: 9] --- Iteration: 34800, Acc: 0.7757142857142857.
[Epoch: 10] --- Iteration: 39000, Loss: 2.0203742220741816.

- MyNet46-4(512)
Finally correct predicted: 0.7585714285714286
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38600, Acc: 0.7714285714285715
[Epoch: 10] --- Iteration: 39000, Loss: 3.484916457338211.



### 2.2.2. 500k(adam,0.0001,e=10) + 10*0.500k(adam,0.0001)

- MyNet44(512)
Final: 0.8157142857142857
Best : [Epoch: 20] --- Iteration: 300, Acc: 0.8364285714285714.

-MyNet44(384)
Final : 0.8285714285714286
Best  : [Epoch: 20] --- Iteration: 200, Acc: 0.8278571428571428


### 2.2.3. 500k(adam,0.0001,e=10) + 10*0.500k(adam,0.001,lineral)
- MyNet42(512)
Finally correct predicted: 0.8057142857142857
Predict one set cost time: 2 s
Best : [Epoch: 52] --- Acc: 0.8128571428571428


### 2.2.4. 500k(adam,0.0001,e=10)-20*0.500k(adam,0.001,lineral)

- MyVGG41(512)
Finally correct predicted: 0.835
Predict one set cost time: 2 s
Best : [Epoch: 28] --- Iteration: 200, Acc: 0.8407142857142857


### 2.2.5. 500k(adam,0.0001,e=10)-50*1bun(adam,0.0001)

- MyNet44(384)
Final : 0.8285714285714286
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8428571428571429

- MyNet44(512)
Final : 0.8335714285714285
Best  : [Epoch: 47] --- Iteration: 100, Acc: 0.8464285714285714

- MyNet46(512)
Final : 0.8278571428571428
Best  : [Epoch: 53] --- Iteration: 100, Acc: 0.8271428571428572


### 2.2.6. 500k(adam,0.0001,e=10)-50*1bun(adam,0.001,lineral)

- MyVGG41(512)
Finally correct predicted: 0.8321428571428572
Predict one set cost time: 2 s
Best : [Epoch: 55] --- Acc: 0.8421428571428572.

- MyNet42(512)
Finally correct predicted: 0.8057142857142857
Predict one set cost time: 2 s
Best : [Epoch: 52] --- Acc: 0.8128571428571428

## 2.3. 网络结构专门对比

* MO-PT-CC-GF-RF-GF-RF-GF-RF-GF-GN 
* 这里的 MyNet 都不再使用 first 33

### 2.3.1. (废弃) 500k + 10 epoch  发现 10 epoch 还不足够充分收敛 
7font

MyNet42
best:     0.7757142857142857
bestiter: 36000
final:    0.7714285714285715  

MyNet44
best:     0.8278571428571428
bestiter: 25700
final:    0.8157142857142857

MyNet46:
best:     0.8171428571428572
bestiter: 37900
final:    0.8092857142857143

MyNet48:
best:     0.8057142857142857
bestiter: 35500
final:    0.7892857142857143


MobileNetV2:  
Finally correct predicted: 0.7928571428571428
Best : [Epoch: 10] --- Iteration: 38800, Acc: 0.8028571428571428

MyVGG
Finally correct predicted: 0.8264285714285714
Predict one set cost time: 2 s
Best : [Epoch: 10] --- Iteration: 38300, Acc: 0.8428571428571429

ResNet34
Finally correct predicted: 0.8228571428571428
Predict one set cost time: 4 s
Best : [Epoch: 10] --- Iteration: 35800, Acc: 0.8435714285714285

-----------------------------------------------------------------------------------

9font 加入自搜的2个手写字体
MO-PT-CC-GF-RF-GF-RF-GF-RF-GF-GN  
500k + 10 epoch  
| 类型网络 | 2                  | 4                  | 6                  | 8                  | vgg                |
| -------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| best     | 0.8057142857142857 | 0.855              | 0.845              | 0.8292857142857143 | 0.855              |
| bestiter | 37800              | 37300              | 30200              | 38300              | 35700              |
| final    | 0.8007142857142857 | 0.8407142857142857 | 0.8314285714285714 | 0.815              | 0.8307142857142857 |

### 2.3.2. (废弃) 500k + 20 epoch 发现20 epoch 仍然有提升空间

MyNet42(LDLD)
Finally correct predicted: 0.7907142857142857
Predict one set cost time: 2 s
Best : [Epoch: 20] --- Iteration: 78000, Acc: 0.7964285714285714. 

MyNet43(LDLD)
Finally correct predicted: 0.8035714285714286
Predict one set cost time: 2 s
Best : [Epoch: 14] --- Iteration: 52000, Acc: 0.8292857142857143

MyNet43(D)
Finally correct predicted: 0.8164285714285714
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 69900, Acc: 0.8285714285714286

MyNet44(LDLD)
Finally correct predicted: 0.8292857142857143
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 71200, Acc: 0.845.

MyNet44(D)
Finally correct predicted: 0.825
Predict one set cost time: 2 s  
Best : [Epoch: 17] --- Iteration: 65300, Acc: 0.8471428571428572.

MyNet44(DLD)
Finally correct predicted: 0.85
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 62900, Acc: 0.8585714285714285.
Finally correct predicted: 0.8442857142857143
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 65900, Acc: 0.8514285714285714.

MyNet45(LDLD)
Finally correct predicted: 0.8042857142857143
Predict one set cost time: 2 s
Best : [Epoch: 17] --- Iteration: 64100, Acc: 0.8257142857142857

MyNet45(D)
Finally correct predicted: 0.8435714285714285
Predict one set cost time: 2 s
Best : [Epoch: 16] --- Iteration: 60800, Acc: 0.85

MyNet45(DLD)
Finally correct predicted: 0.84
Predict one set cost time: 2 s
Best : [Epoch: 20] --- Iteration: 77700, Acc: 0.8542857142857143.

MyNet46(DLD)
Finally correct predicted: 0.8292857142857143
Predict one set cost time: 2 s
Best : [Epoch: 16] --- Iteration: 59600, Acc: 0.8528571428571429

MyNet46(LDLD)
Finally correct predicted: 0.8021428571428572
Predict one set cost time: 2 s
Best : [Epoch: 18] --- Iteration: 66900, Acc: 0.8164285714285714


MyNet47(DLD)
Finally correct predicted: 0.8257142857142857
Predict one set cost time: 2 s
Best : [Epoch: 15] --- Iteration: 55100, Acc: 0.855.

MyNet4(6543)(DLD)
Finally correct predicted: 0.8242857142857143 
Predict one set cost time: 2 s 
2.4. Best : [Epoch: 20] --- Iteration: 75600, Acc: 0.8492857142857143.

-----------------------------------------
MyVGG(LDLD)
Finally correct predicted: 0.8285714285714286
Predict one set cost time: 2 s
Best : [Epoch: 15] --- Iteration: 56300, Acc: 0.8478571428571429.

MyVGG(D)
Finally correct predicted: 0.83
Predict one set cost time: 2 s
Best : [Epoch: 19] --- Iteration: 71100, Acc: 0.845.

------------------------------------------------

MobileNetV2
Finally correct predicted: 0.815
Predict one set cost time: 3 s
Best : [Epoch: 20] --- Iteration: 76100, Acc: 0.8442857142857143.

ResNet34
Finally correct predicted: 0.8357142857142857
Predict one set cost time: 3 s
Best : [Epoch: 14] --- Iteration: 54200, Acc: 0.8464285714285714

### 2.3.3. 500k + 40 epoch

MyVGG(DLD)
Finally top1 accuracy: 0.8621428571428571
Finally top2 accuracy: 0.9092857142857143
Best : [Epoch: 25] --- Iteration: 96700, Acc: 0.8742857142857143.

MyNet44(DLD)
Finally top1 accuracy: 0.8478571428571429
Finally top2 accuracy: 0.9042857142857142
Best : [Epoch: 31] --- Iteration: 120400, Acc: 0.86. 

MyNet43(DLD)
Finally top1 accuracy: 0.8464285714285714
Finally top2 accuracy: 0.9057142857142857
Best : [Epoch: 35] --- Iteration: 136200, Acc: 0.8571428571428571.

MyNet45(DLD)
Finally top1 accuracy: 0.8521428571428571
Finally top2 accuracy: 0.9042857142857142
Best : [Epoch: 39] --- Iteration: 151900, Acc: 0.8592857142857143

MyNet46(DLD)
Finally top1 accuracy: 0.861
Best: 0.869



### 2.3.4. 500K + 相同训练loss

# 3. JPSC1400 Use NNS 120

* 噪点非常重要
* HOG对模糊不敏感
* 随机滤波之后再加噪点效果更好
* 16对噪点非常敏感, GF放在最后对16的精度有提升, 但是对全局精度没好处

* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   

## 3.1. 7fontJpan padding=0

### 3.1.1. NNS

16 : 0.5121428571428571
32 : 0.5414285714285715
64 : 0.5342857142857143

## 3.2. 7fontJpan padding=1
16 : 0.5192857142857142
32 : 0.5371428571428571
64 : 0.5585714285714286
### 3.2.1. NNS

## 3.3. 7fontJpan padding=2


16 : 0.5078571428571429
32 : 0.5257142857142857
64 : 0.5692857142857143

### 3.3.1. 和 Horie 手法比较 种子归一后的数据

CC-GF-RF-GN (种子设定一致)
* 16均值:       0.4899285714285713
* 16ens :       0.6264285714285714
* 32均值 :      0.5903035714285715
* 32ens :       0.665
* 64均值 :      0.561625
* 64ens :       0.6742857142857143
* 120 MSR-3 :   0.7171428571428572

CC(lib20)-GF-RF-GN  (种子设定一致)
* 16均值:       0.5327321428571429 
* 16ens :       0.6507142857142857
* 32均值 :      0.5862499999999998
* 32ens :       0.66
* 64均值 :      0.5587500000000001
* 64ens :       0.6728571428571428
* 120 MSR-3 :   0.7278571428571429

-----------------------------

MO(horie)-CC-GF-RF-GN                   (种子设定一致)
* 16均值:       0.4791964285714285
* 16ens :       0.6521428571428571
* 32均值 :      0.5881607142857144
* 32ens :       0.6985714285714286
* 64均值 :      0.5449285714285714
* 64ens :       0.675
* 120 MSR-3 :   0.7342857142857143

<!-- 
MO(0.7,d4,d8,pU,pD)-CC-GF-RF-GN         (种子设定一致)
* 16均值:       0.4729107142857144
* 16ens :       0.6435714285714286
* 32均值 :      0.5964107142857145
* 32ens :       0.6892857142857143
* 64均值 :      0.5581964285714286
* 64ens :       0.6735714285714286
* 120 MSR-3 :   0.7407142857142858 -->

MO(horie)-CC(lib)-GF-RF-GN              (种子设定一致)
* 16均值:       0.5213035714285715
* 16ens :       0.6742857142857143
* 32均值 :      0.5852678571428569
* 32ens :       0.6914285714285714
* 64均值 :      0.5481964285714286
* 64ens :       0.6842857142857143
* 120 MSR-3 :   0.7414285714285714


MO(0.7,d4,d8,pU,pD)-CC(lib)-GF-RF-GN   (种子设定一致)
* 16均值:       0.5193392857142858
* 16ens :       0.66
* 32均值 :      0.5911607142857144
* 32ens :       0.6878571428571428
* 64均值 :      0.5555892857142857
* 64ens :       0.6764285714285714
* 120 MSR-3 :   0.7364285714285714


MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN   (种子设定一致)
* 16均值:       0.47167857142857156
* 16ens :       0.6471428571428571
* 32均值 :      0.5979107142857143
* 32ens :       0.6928571428571428
* 64均值 :      0.5573392857142857
* 64ens :       0.6678571428571428
* 120 MSR-3 :   0.7342857142857143

----------------------------------------------------------------

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GN   (种子设定一致)
* 16均值:       0.44885714285714295
* 16ens :       0.6614285714285715
* 32均值 :      0.5804107142857142
* 32ens :       0.7085714285714285
* 64均值 :      0.5351964285714284
* 64ens :       0.6878571428571428
* 120 MSR-3 :   0.7471428571428571

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.42716071428571434
* 16ens :       0.6471428571428571
* 32均值 :      0.5867321428571429
* 32ens :       0.7157142857142857
* 64均值 :      0.5428392857142856
* 64ens :       0.7
* 120 MSR-3 :   0.7528571428571429

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GN   (种子设定一致)
* 16均值:       0.4278928571428571
* 16ens :       0.6628571428571428
* 32均值 :      0.5610535714285715
* 32ens :       0.7128571428571429
* 64均值 :      0.5139285714285712
* 64ens :       0.7028571428571428
* 120 MSR-3 :   0.7542857142857143

MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN      (种子设定一致)
* 16均值:       0.41089285714285706
* 16ens :       0.6557142857142857
* 32均值 :      0.5640892857142858
* 32ens :       0.7207142857142858
* 64均值 :      0.5198392857142857
* 64ens :       0.7092857142857143
* 120 MSR-3 :   0.7685714285714286

MO(0.7,d4,d8,pU,pD)-PT(dir,-5,5)-CC(gap:50)-GF-RF-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.34992857142857153
* 16ens :       0.6514285714285715
* 32均值 :      0.4982678571428571
* 32ens :       0.7414285714285714
* 64均值 :      0.45125000000000004
* 64ens :       0.7257142857142858
* 120 MSR-3 :   0.7807142857142857

MO(horie)-PT(dir,-5,5)-CC(lib)-GF-RF-GF-RF-GF-RF-GF-GN   (种子设定一致)
* 16均值:       0.3835535714285713
* 16ens :       0.69
* 32均值 :      0.48410714285714285
* 32ens :       0.735
* 64均值 :      0.43892857142857145
* 64ens :       0.7292857142857143
* 120 MSR-3 :   0.7935714285714286


* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   

* 16均值:       
* 16ens :       
* 32均值 :      
* 32ens :       
* 64均值 :      
* 64ens :       
* 120 MSR-3 :   


--------------------------------------------------------------------------

PT(-0.1,0.2)-CC-GF-RF-GN
* 16均值:     0.39180357142857136
* 16ens :     0.6435714285714286
* 32均值 :    0.5076785714285714
* 32ens :     0.6971428571428572
* 64均值 :    0.47101785714285727
* 64ens :     0.7114285714285714
* 120 MSR-3 : 0.7621428571428571

PT_dir(-5,5)-CC-GF-RF-GN
* 16均值:     0.4165
* 16ens :     0.6564285714285715
* 32均值 :    0.5364464285714287
* 32ens :     0.7185714285714285
* 64均值 :    0.48914285714285705
* 64ens :     0.725
* 120 MSR-3 : 0.7635714285714286



------------------------------------------------
PT_dir(-5,5)-MO(horie)-CC(lib20)-GF-RF-GN
* 16均值:     0.4498214285714286
* 16ens :     0.7107142857142857
* 32均值 :     0.5159464285714284
* 32ens :     0.7292857142857143
* 64均值 :    0.47285714285714275
* 64ens :     0.7214285714285714
* 120 MSR-3 : 0.7771428571428571


PT(-0.1,0.2)-MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN
* 16均值:     0.32730357142857147
* 16ens :     0.6707142857142857
* 32均值 :    0.4996607142857143
* 32ens :     0.725
* 64均值 :    0.4710357142857142
* 64ens :     0.7278571428571429
* 120 MSR-3 : 0.7828571428571428  

Reversed: 
MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.376107142857143
* 16ens :     0.6535714285714286
* 32均值 :    0.5099464285714286
* 32ens :     0.705
* 64均值 :    0.47401785714285716
* 64ens :     0.7285714285714285
* 120 MSR-3 : 0.7664285714285715
------------------------------------------------------


MO(0.7,d4,d8,pU,pD)-PT_dir(-5,5)-CC(gap:50)-GF-RF-GN
* 16均值:     0.3997500000000001
* 16ens :     0.6735714285714286
* 32均值 :    0.5379821428571427
* 32ens :     0.7242857142857143
* 64均值 :    0.4885357142857143
* 64ens :     0.7228571428571429
* 120 MSR-3 : 0.7735714285714286

MO(horie)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.3756785714285714
* 16ens :     0.66
* 32均值 :    0.49557142857142866
* 32ens :     0.7214285714285714
* 64均值 :    0.4543750000000001
* 64ens :     0.7242857142857143
* 120 MSR-3 : 0.7721428571428571

MO(horie)-PT_dir(-5,5)-CC(gap:50)-GF-RF-GN
* 16均值:     0.40514285714285714
* 16ens :     0.6885714285714286
* 32均值 :    0.5240714285714286
* 32ens :     0.7278571428571429
* 64均值 :    0.47648214285714274
* 64ens :     0.73
* 120 MSR-3 : 0.7778571428571428

### 3.3.2. 少变换

CC
* 16均值: 0.5082321428571429
* 16ens :  0.5085714285714286
* 32均值 : 0.5293928571428571
* 32ens :  0.5307142857142857
* 64均值 : 0.5705178571428572
* 64ens : 0.5757142857142857
* 120 MSR-3 : 0.5821428571428572

CC-GN
* 16均值: 0.5229821428571428
* 16ens : 0.5835714285714285
* 32均值 : 0.5901071428571428
* 32ens : 0.6121428571428571
* 64均值 : 0.5714821428571428
* 64ens : 0.6142857142857143
* 120 MSR-3 : 0.6607142857142857

CC-RF-GN
* 16均值: 0.5163749999999998
* 16ens : 0.6107142857142858
* 32均值 : 0.5778392857142857
* 32ens : 0.6442857142857142
* 64均值 : 0.5580178571428569
* 64ens : 0.6564285714285715
* 120 MSR-3 : 0.705

### 3.3.3. 四标准变化单RF

CC-GF-RF-GN
* 16均值: 0.49169642857142853
* 16ens : 0.6307142857142857
* 32均值 : 0.5879107142857142
* 32ens : 0.66
* 64均值 : 0.5593749999999998
* 64ens : 0.67
* 120 MSR-3 : 0.7235714285714285
* 120 MSR-3 : 0.7228571428571429 (第二次)

CC(lib20)-GF-RF-GN
* 16均值: 0.5299285714285713
* 16ens : 0.6464285714285715
* 32均值 : 0.5845178571428572
* 32ens : 0.6685714285714286
* 64均值 : 0.557392857142857
* 64ens : 0.6728571428571428
* 120 MSR-3 : 0.7185714285714285

CC-GF-RF-GN-GF
* 16均值: 0.5853750000000001
* 16ens : 0.6528571428571428
* 32均值 : 0.6016250000000001
* 32ens : 0.6664285714285715
* 64均值 : 0.5709107142857145
* 64ens : 0.6764285714285714
* 120 MSR-3 : 0.7214285714285714  反而还降了，

CC-GF-RF-GF-GN
* 16均值: 0.4588214285714285
* 16ens : 0.6235714285714286
* 32均值 : 0.5991428571428573
* 32ens : 0.6778571428571428
* 64均值 : 0.5685535714285714
* 64ens : 0.685
* 120 MSR-3 : 0.7321428571428571

CC-GF-RF-GN-GF-GN-GF
* 16均值:  0.5751071428571428
* 16ens : 0.6664285714285715
* 32均值 : 0.6082499999999998
* 32ens : 0.6864285714285714
* 64均值 : 0.5711964285714288
* 64ens : 0.685
* 120 MSR-3 : 0.7271428571428571


### 3.3.4. 四标准多RF

CC-GF-RF-GF-RF-GF-RF-GF-GN

CC-GF-RF-GF-RF-GF-GN
* 16均值: 0.4432142857142859
* 16ens : 0.6521428571428571
* 32均值 : 0.5862142857142858
* 32ens : 0.7157142857142857
* 64均值 : 0.5513571428571428
* 64ens : 0.7028571428571428
* 120 MSR-3 :  0.7535714285714286

CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.4226964285714286
* 16ens : 0.655
* 32均值 : 0.56625
* 32ens : 0.7192857142857143
* 64均值 : 0.5238571428571428
* 64ens :  0.7078571428571429 
* 120 MSR-3 : 0.7578571428571429

CC-GF-RF-GF-RF-GF-RF-GF-GN (2)
* 16均值: 0.42312500000000003
* 16ens : 0.6628571428571428
* 32均值 : 0.5685714285714285
* 32ens : 0.7207142857142858
* 64均值 : 0.528357142857143
* 64ens : 0.71
* 120 MSR-3 : 0.7621428571428571



### 3.3.5. 加入单几何或形态学

-----------------------MO-------------------------

MO(p=0.3, d4,d8,pU)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.42207142857142854
* 16ens : 0.6714285714285714
* 32均值 : 0.5697321428571427
* 32ens : 0.7271428571428571
* 64均值 : 0.523857142857143
* 64ens : 0.7085714285714285
* 120 MSR-3 : 0.7685714285714286

MO(p=0.3, d4,d8,pU)-CC-GF-RF-GF-RF-GF-RF-GF-GN (2)
* 16均值: 0.42207142857142854
* 16ens : 0.6714285714285714
* 32均值 : 0.5697321428571427
* 32ens : 0.7271428571428571
* 64均值 : 0.523857142857143
* 64ens : 0.7085714285714285
* 120 MSR-3 : 0.7685714285714286

---------------------------AF --------------------------------

AF(p=1, ro=20, sh=(10,10))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.28858928571428577
* 16ens : 0.6257142857142857
* 32均值 : 0.4144821428571429
* 32ens : 0.6992857142857143
* 64均值 : 0.373482142857143
* 64ens : 0.685
* 120 MSR-3 : 0.755

 
AF(p=1, ro=20, sh=(15,15))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.27005357142857145
* 16ens : 0.6264285714285714
* 32均值 : 0.3912321428571428
* 32ens : 0.695
* 64均值 : 0.35505357142857147
* 64ens : 0.6971428571428572
* 120 MSR-3 : 0.7592857142857142

AF(p=1, ro=30, sh=(10,10))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.24335714285714288
* 16ens : 0.5835714285714285
* 32均值 : 0.36321428571428577
* 32ens : 0.6907142857142857
* 64均值 : 0.31519642857142854
* 64ens :  0.67
* 120 MSR-3 : 0.7464285714285714

AF(p=1, ro=30, sh=(15,15))-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.22530357142857146
* 16ens : 0.5657142857142857
* 32均值 : 0.33825
* 32ens :  0.6578571428571428
* 64均值 : 0.30189285714285713
* 64ens :  0.6621428571428571
* 120 MSR-3 : 0.7321428571428571

-----------------------------------------PT--------------------------

PT(0,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.38489285714285715
* 16ens : 0.665
* 32均值 : 0.5291428571428571
* 32ens : 0.7135714285714285
* 64均值 : 0.5013928571428572
* 64ens :  0.7114285714285714
* 120 MSR-3 : 0.7685714285714286

PT(0,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3215892857142858
* 16ens : 0.615 
* 32均值 : 0.45326785714285717
* 32ens : 0.6921428571428572
* 64均值 : 0.42474999999999985
* 64ens : 0.6764285714285714
* 120 MSR-3 : 0.745

PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3964285714285714
* 16ens : 0.6664285714285715
* 32均值 : 0.5350535714285714
* 32ens : 0.7228571428571429
* 64均值 : 0.4935892857142858
* 64ens : 0.7278571428571429
* 120 MSR-3 : 0.7742857142857142

PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.3322142857142857
* 16ens : 0.6571428571428571
* 32均值 : 0.47275
* 32ens : 0.7228571428571429
* 64均值 : 0.434875
* 64ens :  0.7128571428571429
* 120 MSR-3 : 0.77
 
### 3.3.6. 形态学+投影变换


MO(p=0.3, d4,d8,pU)-PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :

MO(p=0.7, d4,d8,pU)-PT(-0.1,0.1)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.38658928571428575
* 16ens : 0.6757142857142857
* 32均值 : 0.5393035714285717
* 32ens : 0.7314285714285714
* 64均值 : 0.48953571428571435
* 64ens :  0.7157142857142857
* 120 MSR-3 : 0.7792857142857142


MO(p=0.3, d4,d8,pU)-PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.32866071428571425
* 16ens : 0.6678571428571428
* 32均值 : 0.4697321428571429
* 32ens : 0.7307142857142858
* 64均值 : 0.4373214285714284
* 64ens :  0.7292857142857143
* 120 MSR-3 : 0.7821428571428571

MO(p=0.7, d4,d8,pU)-PT(-0.1,0.2)-CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值:  0.3263214285714286
* 16ens :  0.6578571428571428
* 32均值 : 0.4781607142857144
* 32ens :  0.735
* 64均值 : 0.43808928571428574
* 64ens :  0.7171428571428572
* 120 MSR-3 : 0.7821428571428571

--------------------------------------------

PT(-0.1,0.2)-MO(0.7,d4,d8,pU,pD)-CC(gap:50)-GF-RF-GN
* 16均值:     0.32730357142857147
* 16ens :     0.6707142857142857
* 32均值 :    0.4996607142857143
* 32ens :     0.725
* 64均值 :    0.4710357142857142
* 64ens :     0.7278571428571429
* 120 MSR-3 : 0.7828571428571428  

Reversed: 
MO(0.7,d4,d8,pU,pD)-PT(-0.1,0.2)-CC(gap:50)-GF-RF-GN
* 16均值:     0.376107142857143
* 16ens :     0.6535714285714286
* 32均值 :    0.5099464285714286
* 32ens :     0.705
* 64均值 :    0.47401785714285716
* 64ens :     0.7285714285714285
* 120 MSR-3 : 0.7664285714285715



## 3.4. 7fontJpan revised  3097

## 3.5. 7fontJpan padding=2 revised  3104　ヘぺべ　已舍弃 

### 3.5.1. NNS

* 16 ：0.51
* 32 ：0.5278571428571428
* 64 ：0.5707142857142857

* BaseLine : 57



#### 3.5.1.1. 单变换 baseline

颜色 gap 50:
* 16均值:   0.5107857142857143
* 16ens :  0.5107142857142857
* 32均值 : 0.5310535714285713
* 32ens :  0.5328571428571428
* 64均值 :  0.5730357142857144
* 64ens :  0.575
* 120 MSR-3 : 0.5707142857142857

#### 3.5.1.2. 双变换 

Color(gap:50)-GF(sigma: 0-10)
* 16均值: 0.5099107142857142
* 16ens : 0.515
* 32均值 : 0.5586785714285714 
* 32ens : 0.5607142857142857
* 64均值 : 0.5843214285714285
* 64ens : 0.5892857142857143 
* 120 MSR-3 : 0.5907142857142857 

Color(gap:50)-RF(scale(-1,1))
* 16均值: 0.4855714285714286
* 16ens : 0.5307142857142857
* 32均值 : 0.5233214285714285
* 32ens : 0.5828571428571429
* 64均值 : 0.5510892857142857
* 64ens : 0.6414285714285715
* 120 MSR-3 : 0.6492857142857142
 

Color(gap:50)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.5226964285714284
* 16ens : 0.5807142857142857
* 32均值 : 0.5926607142857143
* 32ens : 0.6164285714285714
* 64均值 : 0.5724464285714286
* 64ens : 0.6221428571428571
* 120 MSR-3 : 0.665
 

#### 3.5.1.3. 三变化

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))
* 16均值: 0.5035357142857142
* 16ens : 0.5564285714285714
* 32均值 : 0.5451964285714286 
* 32ens :0.615
* 64均值 : 0.5567678571428571 
* 64ens : 0.6457142857142857
* 120 MSR-3 : 0.6721428571428572

Color(Gap:50)-RF(scale(-1,1))-GF(sigma: 0-10)
* 16均值:  0.5038571428571429
* 16ens : 0.5521428571428572
* 32均值 : 0.5497321428571429
* 32ens : 0.6071428571428571
* 64均值 :  0.5678392857142858
* 64ens :  0.6492857142857142
* 120 MSR-3 :  0.67


Color(Gap:50)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.5195000000000001
* 16ens : 0.6192857142857143
* 32均值 :0.5806071428571429
* 32ens : 0.645
* 64均值 : 0.558875
* 64ens : 0.6628571428571428
* 120 MSR-3 : 0.7078571428571429


Color(Gap:50)-GN(mean: 0, var 0.01,0.01)-RF(scale(-1,1))
* 16均值: 0.4343571428571429
* 16ens : 0.5992857142857143
* 32均值 : 0.5541607142857142
* 32ens : 0.6457142857142857
* 64均值 : 0.5456071428571428
* 64ens : 0.655
* 120 MSR-3 : 0.7085714285714285

#### 3.5.1.4. 四标准变换 


Color(Gap:50)-GN(mean: 0, var 0.01,0.01)-RF(scale(-1,1))-GF(sigma: 0-10)
* 16均值: 0.5398214285714286 
* 16ens : 0.6457142857142857
* 32均值 : 0.5736964285714286
* 32ens : 0.6571428571428571
* 64均值 : 0.5575178571428572 
* 64ens :  0.6735714285714286
* 120 MSR-3 : 0.7107142857142857


Color(Gap:50)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值:  0.582
* 16ens : 0.6528571428571428
* 32均值 : 0.5973928571428573
* 32ens : 0.6635714285714286
* 64均值 : 0.5707857142857142 
* 64ens : 0.6678571428571428
* 120 MSR-3 : 0.7114285714285714

Color(Gap:50)-RF(scale(-1,1))-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.477625
* 16ens : 0.6035714285714285
* 32均值 : 0.5946964285714286
* 32ens : 0.6614285714285715
* 64均值 : 0.5690714285714287
* 64ens : 0.6707142857142857
* 120 MSR-3 : 0.72s

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)
* 16均值:  0.4923571428571429 
* 16ens : 0.6321428571428571
* 32均值 : 0.5932321428571428
* 32ens : 0.6671428571428571
* 64均值 : 0.5606071428571427
* 64ens : 0.6735714285714286
* 120 MSR-3 : 0.7264285714285714

#### 3.5.1.5. 四标准加入重复的- 单RF

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5893928571428575
* 16ens : 0.6642857142857143
* 32均值 : 0.606107142857143
* 32ens : 0.685
* 64均值 : 0.5713749999999999 
* 64ens : 0.6871428571428572
* 120 MSR-3 : 0.7285714285714285

Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值: 0.42957142857142855
* 16ens : 0.6035714285714285
* 32均值 : 0.5930892857142859
* 32ens : 0.6764285714285714
* 64均值 : 0.5651428571428572
* 64ens : 0.6878571428571428
* 120 MSR-3 : 0.7364285714285714
 
Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5752857142857144
* 16ens :  0.6721428571428572
* 32均值 : 0.6070178571428572
* 32ens : 0.6871428571428572
* 64均值 : 0.5744285714285713
* 64ens : 0.695 
* 120 MSR-3 : 0.74


#### 3.5.1.6. 四标准加入重复的- 多RF


Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)
* 16均值:  0.44135714285714284 
* 16ens : 0.6535714285714286
* 32均值 :  0.5850892857142858
* 32ens : 0.7114285714285714
* 64均值 :   0.5520535714285716
* 64ens : 0.7071428571428572
* 120 MSR-3 : 0.7557142857142857


Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)-RF(scale(-1,1))-GN(mean: 0, var 0.01,0.01)-GF(sigma: 0-10)
* 16均值: 0.5358392857142856
* 16ens : 0.6842857142857143
* 32均值 : 0.5775178571428572
* 32ens : 0.7042857142857143
* 64均值 : 0.5383392857142857
* 64ens : 0.6978571428571428
* 120 MSR-3 : 0.7471428571428571


CC-GF-RF-GF-RF-GF-RF-GF-GN
* 16均值: 0.42751785714285706
* 16ens : 0.675
* 32均值 : 0.5695178571428572
* 32ens : 0.72
* 64均值 : 0.5253392857142857
* 64ens : 0.7135714285714285
* 120 MSR-3 : 0.7592857142857142


#### 3.5.1.7. 加入几何

PT(p:1, 0-0,3)-Color(Gap:50)-GF(sigma: 0-10)-RF(-1,1)-GN(mean:0, var: 0.01-0.01):
* 16均值: 0.2902321428571429
* 16ens : 0.5514285714285714
* 32均值 : 0.3886071428571428
* 32ens : 0.6135714285714285
* 64均值 : 0.35160714285714273
* 64ens : 0.5964285714285714
* 120 MSR-3 : 0.6921428571428572

AT(p:1, rotation: 20, shear: x15, y15)-Color(Gap:50)-GF(sigma: 0-10)-RF(-1,1)-GN(mean:0, var: 0.01-0.01):
* 16均值:　0.3181785714285714
* 16ens :　0.5871428571428572
* 32均值 :　0.42912499999999987
* 32ens :　0.6607142857142857
* 64均值 :　0.38626785714285716
* 64ens :　0.6671428571428571
* 120 MSR-3 :　0.7314285714285714


* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :


* 16均值:
* 16ens :
* 32均值 :
* 32ens :
* 64均值 :
* 64ens :
* 120 MSR-3 :


### 3.5.2. Deepnetwork


#### 3.5.2.1. 200,000

* 64batch size
* 20epoch

AT(p:1, rotation: 20, shear: x15, y15)-Color(Gap:50)-RF(-1,1)-GN(mean:0, var: 0.01-0.01)-GF(sigma: 0-10):
* MyNet 44 0.5807142857142857 
  * Best : [Epoch: 18] --- Iteration: 54900, Acc: 0.595.
* MyNet 64 收敛失败
* MyNet 84 收敛失败
* ------------------
* Alexnet : 收敛失败
* VGG11 ：收敛失败
* Resnet34 : 0.5814285714285714
  * Best : [Epoch: 20] --- Iteration: 61400, Acc: 0.6121428571428571.




PT(p:1, 0-0,3)-Color(Gap:50)-RF(-1,1)-GN(mean:0, var: 0.01-0.01)-GF(sigma: 0-10)
* MyNet 44 0.5807142857142857 
  * Best : [Epoch: 20] --- Iteration: 61100, Acc: 0.6435714285714286.
* MyNet 64 0.4642857142857143
  * Best : [Epoch: 19] --- Iteration: 56700, Acc: 0.5328571428571428.
* MyNet 84 收敛失败



#### 3.5.2.2. 500,000
* 64batch size
* 20epoch
* Resnet34 : 0.7857142857142857
* MyNet 44 : 0.6685714285714286
  * Best :  [Epoch: 15] --- Iteration: 113200, Acc: 0.6921428571428572.
* MyNet 64 
  * Best : 
* MyNet 84 

PT(p:1, 0-0,3)-Color(Gap:50)-GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-RF(scale(-1,1))--GF(sigma: 0-10)-GN(mean: 0, var 0.01,0.01)

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

* MyNet 44 
  * Best : 
* MyNet 64 
  * Best : 
* MyNet 84 

