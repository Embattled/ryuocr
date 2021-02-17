from matplotlib import pyplot as plt


x = range(1,201)

y = [
    8.036604387619915,
    8.020883173101089,
    7.993359470367432,
    7.9553744400248805,
    7.9097277506660015,
    7.8585720118354345,
    7.804946890598585,
    7.75259947496302,
    7.7040952451867994,
    7.660400432137882,
    7.621406853517747,
    7.5866589003918214,
    7.5556352839750405,
    7.527834074437117,
    7.502818908691406,
    7.480214568446664,
    7.459700510823603,
    7.4410066860174044,
    7.423906283481941,
    7.408206651070539,
    7.393743786691617,
    7.380378952128364,
    7.367993951880414,
    7.35648267362632,
    7.3457550544738766,
    7.335735015092392,
    7.326353724485909,
    7.3175512053385505,
    7.309275794657926,
    7.301481341193704,
    7.294127446560072,
    7.2871770455556755,
    7.280598026758422,
    7.274361873250519,
    7.26844152274252,
    7.262814093882741,
    7.2574576928240315,
    7.252353677365802,
    7.2474845333876114,
    7.242834072814269,
    7.238387880229539,
    7.23413298310352,
    7.2300569690855685,
    7.226149137899838,
    7.222398694356283,
    7.218796517477011,
    7.215334089706478,
    7.212003295678718,
    7.208796909419285,
    7.2057078226874856,
    7.20272986908135,
    7.199857151993799,
    7.197084041070462,
    7.194405515209522,
    7.191816856465875,
    7.189313520603821,
    7.186891338478301,
    7.1845464032271815,
    7.182275145503127,
    7.180074185764088,
    7.17794021111675,
    7.175870261020407,
    7.173861564508005,
    7.171911272055962,
    7.170016948820779,
    7.16817630778022,
    7.166387045477649,
    7.164646946022668,
    7.162954115156221,
    7.161306654825932,
    7.159702734247949,
    7.15814070802888,
    7.156618877460071,
    7.155135790608077,
    7.153689958609787,
    7.152280015915909,
    7.150904580634272,
    7.149562489752676,
    7.1482525505242025,
    7.146973577878055,
    7.145724504064977,
    7.144504286055928,
    7.14331196278735,
    7.142146581895545,
    7.141007255039413,
    7.139893132782504,
    7.138803387397524,
    7.137737182563639,
    7.136693839245258,
    7.135672615961312,
    7.134672781137319,
    7.13369370622708,
    7.13273475014809,
    7.131795296412386,
    7.1308747844282685,
    7.129972647159708,
    7.129088329039898,
    7.128221299067265,
    7.127371078520677,
    7.126537211025463,
    7.125719185130986,
    7.124916583471507,
    7.124128989530658,
    7.123355971472296,
    7.122597129939317,
    7.121852091970243,
    7.12112047621154,
    7.120401939743225,
    7.1196961303048045,
    7.119002715381072,
    7.118321383586938,
    7.1176518230879005,
    7.1169937330878446,
    7.116346821573374,
    7.115710814834555,
    7.115085442941522,
    7.114470428219388,
    7.113865534638836,
    7.113270519018998,
    7.112685133943371,
    7.112109158580374,
    7.111542368165092,
    7.110984553086706,
    7.11043548688038,
    7.109894986635096,
    7.109362837803019,
    7.108838856800003,
    7.108322870687527,
    7.107814690189388,
    7.1073141445833095,
    7.106821061251246,
    7.106335270298569,
    7.105856625330443,
    7.10538496728316,
    7.1049201479145125,
    7.104462018062499,
    7.104010433780118,
    7.1035652610245466,
    7.103126362708984,
    7.1026936136373955,
    7.102266883372664,
    7.10184604523097,
    7.101430985565687,
    7.101021586796817,
    7.10061773242138,
    7.100219314123334,
    7.099826221114972,
    7.0994383498860465,
    7.0990555974461085,
    7.098677865795061,
    7.09830505997465,
    7.097937082506186,
    7.097573840732713,
    7.097215245715346,
    7.096861213633413,
    7.096511655982983,
    7.0961664874797545,
    7.095825626063684,
    7.095488996149388,
    7.095156519448056,
    7.094828117176874,
    7.094503714264349,
    7.094183246395038,
    7.093866637173821,
    7.093553819333381,
    7.093244725301028,
    7.092939287778902,
    7.0926374475495155,
    7.09233913627436,
    7.0920442948786855,
    7.091752865013487,
    7.0914647887734805,
    7.091180010402904,
    7.090898469592204,
    7.090620115488517,
    7.0903448921155166,
    7.090072749199048,
    7.089803634947369,
    7.089537501107929,
    7.089274298007192,
    7.089013978959989,
    7.088756495533101,
    7.088501801884699,
    7.088249856431771,
    7.08800061294119,
    7.087754027199549,
    7.087510059779078,
    7.08726866773431,
    7.08702981431203,
    7.086793457804819,
    7.086559559405309,
    7.08632807866031,
    7.086098982553153,
    7.0858722321838234,
    7.0856477939669,
    7.0854256290776965,
    7.085205706826748,
    7.084987991449738,
    7.084772451932424,
    7.0845590551600734]


plt.xlabel("Training Epoch")
plt.ylabel("Mean Loss")

plt.grid()
plt.plot(x,y)



plt.show()