import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Replace random data with provided vector data for ada-002
vector_data_ada_002 = [
                0.002253932,
                -0.009333183,
                0.01574578,
                -0.007790351,
                -0.004711035,
                0.014844206,
                -0.009739526,
                -0.03822161,
                -0.0069014765,
                -0.028723348,
                0.02523134,
                0.01814574,
                -0.003650735,
                -0.025498003,
                0.0004960238,
                -0.016317198,
                0.028418591,
                0.0053268983,
                0.009618893,
                -0.01644418,
                -0.015339436,
                0.0042634234,
                0.0070284586,
                -0.007079251,
                -0.003920572,
                0.01850129,
                0.008704622,
                -0.02275519,
                0.011447435,
                0.02382184,
                0.015529909,
                -0.0035269274,
                -0.034920074,
                -0.00414914,
                -0.026107516,
                -0.021523464,
                -0.0057459394,
                0.011758541,
                0.008317326,
                0.0040951725,
                0.019174295,
                -0.014437864,
                0.00899668,
                0.006301486,
                -0.04571355,
                0.017828286,
                -0.005501499,
                -0.0007995903,
                -0.022005996,
                -0.0038126372,
                0.020990139,
                -0.017460037,
                -0.0117331445,
                -0.022602811,
                0.016431483,
                0.017193375,
                -0.0085014505,
                0.0015864824,
                0.025078962,
                -0.024964679,
                0.0077522565,
                0.0058221286,
                -0.02218377,
                0.0030475701,
                -0.0061840275,
                -0.025383718,
                -0.0080443155,
                0.001172997,
                0.000031745523,
                0.0046538934,
                0.020672685,
                0.013510894,
                0.0046602427,
                -0.015936252,
                0.016545765,
                -0.008939539,
                -0.0075363866,
                0.013625178,
                -0.0070411568,
                0.0053713424,
                0.0098728575,
                -0.04586593,
                0.0030285227,
                0.023999615,
                0.022945663,
                0.0070856004,
                -0.023618668,
                0.009929999,
                -0.006571323,
                -0.03324391,
                -0.0025904346,
                0.0198473,
                0.0017840983,
                0.0010872841,
                -0.022704396,
                0.0050094435,
                0.015377531,
                0.03161854,
                -0.005415786,
                -0.016063234,
                -0.0050951564,
                0.019859998,
                -0.0094157215,
                -0.0067237015,
                -0.031212198,
                -0.0094474675,
                -0.015212454,
                -0.028824933,
                0.020990139,
                -0.01829812,
                -0.0029682063,
                0.012704558,
                0.005203091,
                -0.049497616,
                -0.04304693,
                -0.00038412082,
                0.021459972,
                -0.016812429,
                0.0065459264,
                -0.041421555,
                0.00096585747,
                0.03352327,
                0.0131045515,
                -0.010482371,
                -0.0004876906,
                0.018425101,
                0.00008020704,
                -0.013091853,
                0.011828382,
                0.0058602234,
                0.0071109966,
                0.008539545,
                0.018425101,
                -0.00070673466,
                -0.019085407,
                0.021891711,
                -0.029739205,
                -0.0042792964,
                0.0014475958,
                -0.005101505,
                0.015034679,
                0.021294896,
                -0.018818745,
                0.01106014,
                -0.007434801,
                0.022526622,
                0.019237787,
                0.014983886,
                -0.003571371,
                0.006288788,
                0.025624985,
                -0.027758284,
                0.03664703,
                -0.0044570714,
                0.013815651,
                0.007987173,
                -0.0056316555,
                0.010317295,
                -0.005818954,
                -0.008691924,
                0.010691891,
                -0.000049552775,
                0.032177262,
                -0.023771046,
                -0.0024809125,
                0.030272529,
                0.029612223,
                0.01650767,
                0.0021301245,
                0.0009452229,
                -0.011574417,
                0.018564781,
                -0.0075871795,
                0.0075236885,
                -0.0011047442,
                0.0045459587,
                0.00992365,
                -0.007644322,
                -0.006291962,
                -0.005799907,
                -0.028266212,
                0.0043491363,
                0.029231276,
                0.024939282,
                -0.01565689,
                -0.018196533,
                0.0076125762,
                0.008641131,
                -0.011688701,
                -0.006352279,
                0.0020523479,
                0.036900993,
                -0.002444405,
                -0.020825062,
                -0.6895635,
                -0.019580638,
                0.0036189894,
                0.006691956,
                0.02829161,
                0.023402799,
                0.0072125825,
                0.010520466,
                0.009606195,
                -0.009587147,
                -0.021282198,
                0.000058133985,
                0.019720318,
                0.0021952027,
                -0.009510959,
                -0.002069808,
                -0.00033729617,
                0.0066475123,
                -0.04119299,
                0.011225216,
                -0.013422007,
                0.02778368,
                -0.011377595,
                0.0001462278,
                0.015402927,
                -0.0005051506,
                0.014412466,
                -0.008552244,
                -0.018907633,
                -0.017688604,
                -0.009377627,
                0.02459643,
                0.012837889,
                -0.008831604,
                0.03502166,
                -0.0051491237,
                -0.024228182,
                0.024139294,
                -0.0037015278,
                0.030983629,
                -0.015009282,
                -0.023263117,
                0.030094754,
                0.0068252874,
                0.010095076,
                0.0114728315,
                0.030323323,
                0.01758702,
                0.0021015536,
                -0.006730051,
                0.009733177,
                0.016634652,
                0.0035904185,
                0.027910663,
                0.004069776,
                0.0030840775,
                0.029789997,
                -0.016774334,
                0.00217933,
                0.02317423,
                0.011879174,
                0.01424739,
                -0.019656828,
                -0.02948524,
                -0.0015912442,
                0.018272722,
                -0.0027602732,
                0.019161597,
                0.027580509,
                -0.008457007,
                0.012368055,
                0.0028317005,
                -0.005225313,
                0.0064729117,
                0.009396674,
                0.02253932,
                0.023517082,
                0.0009928412,
                0.01899652,
                -0.0029301117,
                -0.0069205235,
                -0.0101331705,
                -0.007930031,
                -0.0072506773,
                0.020571098,
                0.010825223,
                -0.029891582,
                -0.0038539064,
                0.0026999565,
                -0.01885684,
                0.029891582,
                0.024634525,
                -0.0044316747,
                0.026412275,
                0.017561622,
                0.0411168,
                -0.020190151,
                0.0147045255,
                0.021510765,
                -0.029942377,
                -0.0014904522,
                0.00340312,
                0.034132786,
                0.03608831,
                0.028037645,
                0.018387007,
                0.0031697904,
                0.009326834,
                0.020685382,
                -0.012501386,
                0.017066393,
                0.011587115,
                -0.007631623,
                0.013193439,
                -0.0073268665,
                -0.022577414,
                0.0059300633,
                0.025777364,
                -0.019034615,
                -0.018971123,
                -0.013282326,
                -0.0026491638,
                0.022437735,
                -0.00794273,
                -0.00076903525,
                0.028190022,
                0.01389184,
                -0.019961584,
                -0.021040931,
                -0.002955508,
                0.013510894,
                0.015974347,
                0.011999807,
                -0.016875919,
                0.0048602396,
                -0.012933126,
                0.025929742,
                -0.030323323,
                0.017561622,
                -0.022437735,
                -0.01885684,
                0.0019396513,
                -0.006082442,
                0.0020952043,
                -0.0015618796,
                -0.011231566,
                -0.005374517,
                -0.011383944,
                -0.021523464,
                0.0051205526,
                0.005590386,
                -0.00022420274,
                -0.012622019,
                0.003396771,
                0.02459643,
                -0.0019698096,
                0.0013031537,
                -0.005485626,
                -0.024748808,
                -0.03055189,
                -0.01594895,
                0.016672747,
                -0.013345817,
                -0.008628433,
                0.009441118,
                -0.0074728956,
                0.0070729023,
                -0.005663401,
                -0.012126789,
                -0.023783745,
                0.028799538,
                -0.015923554,
                -0.003339629,
                0.0053459457,
                -0.016812429,
                0.0014928331,
                -0.021891711,
                -0.019479051,
                -0.0063332315,
                0.016469577,
                0.008755415,
                0.0040412047,
                -0.013879142,
                -0.012914078,
                0.009530006,
                0.005742765,
                0.0014372785,
                0.025256736,
                -0.017815586,
                0.015377531,
                0.00209203,
                -0.010971252,
                0.0073903576,
                0.0014531512,
                0.015987044,
                0.013307722,
                -0.0026571,
                0.0020650462,
                0.014133106,
                -0.0022682175,
                -0.004196758,
                -0.0065840213,
                0.003139632,
                -0.0047967485,
                0.013269628,
                -0.017523527,
                -0.011091885,
                -0.029612223,
                0.0038507318,
                0.007504641,
                0.016063234,
                -0.013739462,
                -0.017269563,
                -0.0038126372,
                0.02140918,
                0.016990203,
                0.01921239,
                0.006869731,
                -0.013942633,
                -0.009199852,
                -0.0023396448,
                -0.02091395,
                -0.0052316617,
                -0.0015706096,
                0.0064252936,
                0.0030602682,
                0.0030570938,
                -0.00730147,
                0.0012237899,
                -0.0021428226,
                -0.0020158407,
                0.018602876,
                0.0007110997,
                0.015085472,
                0.0069649676,
                0.0013475973,
                0.016342595,
                -0.008780811,
                0.016748937,
                -0.0005257852,
                -0.0077141616,
                0.016761634,
                -0.0008011776,
                -0.027910663,
                0.0108506195,
                5.3632573e-7,
                -0.006031649,
                -0.0112442635,
                -0.027453527,
                0.008901444,
                0.012926776,
                0.016812429,
                -0.017612414,
                0.014653733,
                -0.0040221578,
                0.0080125695,
                -0.0018698112,
                -0.018793348,
                0.034056596,
                0.021040931,
                0.01644418,
                0.022272658,
                0.015517211,
                -0.0010134758,
                0.0044761184,
                0.00059364125,
                -0.0076887654,
                -0.026209103,
                0.009764923,
                -0.0067617963,
                0.014793413,
                -0.025917044,
                -0.0023364704,
                -0.0066602104,
                0.0066983053,
                -0.0012110916,
                -0.01864097,
                0.0022459957,
                -0.017904473,
                0.0048062718,
                0.0031618539,
                -0.021790126,
                0.029891582,
                0.011866476,
                -0.0027618604,
                -0.01098395,
                -0.01737115,
                0.0074094045,
                -0.012469641,
                0.003225345,
                0.0072506773,
                0.01438707,
                0.0011595052,
                -0.009587147,
                0.01410771,
                0.0006202281,
                0.046119895,
                -0.02077427,
                0.031567745,
                -0.005145949,
                0.0065840213,
                -0.025193246,
                -0.028113835,
                -0.023517082,
                0.026259895,
                -0.013091853,
                -0.022285355,
                -0.00055792753,
                0.023364704,
                -0.007745907,
                0.0065332283,
                -0.0059649837,
                -0.012507736,
                -0.0021269498,
                0.023834538,
                -0.0033015343,
                -0.029409051,
                0.0019079058,
                0.007720511,
                -0.0008737955,
                -0.030450305,
                -0.016977506,
                -0.008799858,
                -0.013422007,
                0.07811938,
                -0.004914207,
                0.015187058,
                0.008387167,
                0.036621634,
                0.000517452,
                -0.03890731,
                -0.0047078608,
                -0.0054602297,
                -0.01212044,
                -0.0067110034,
                0.009517307,
                0.024939282,
                0.007206233,
                0.026336085,
                -0.0042570746,
                -0.0059427614,
                -0.024507543,
                0.019275881,
                -0.023872633,
                -0.008234789,
                0.016647352,
                0.015936252,
                0.027097978,
                -0.015187058,
                0.0044538965,
                0.025599588,
                0.02615831,
                -0.0011285533,
                -0.023504384,
                0.0016491798,
                -0.0012182344,
                -0.0048824614,
                -0.0022094883,
                -0.015568004,
                0.0048665884,
                0.0005083252,
                -0.012241073,
                0.0048824614,
                -0.0062697404,
                0.040431097,
                -0.0085331965,
                0.0050348397,
                -0.01707909,
                0.0047523044,
                -0.0016571162,
                -0.027605906,
                0.05081823,
                -0.030374115,
                -0.0028904297,
                0.015466418,
                0.027123373,
                0.002236472,
                -0.010545862,
                -0.004311042,
                0.011187122,
                -0.0067110034,
                -0.024228182,
                -0.013968029,
                0.016355293,
                -0.014717224,
                -0.03768828,
                -0.005501499,
                -0.0006134822,
                -0.025993234,
                -0.020380626,
                -0.012660114,
                -0.012507736,
                -0.0029650317,
                -0.0029586826,
                -0.012342659,
                0.0037967644,
                -0.012844238,
                -0.0070411568,
                0.017383847,
                -0.0024269451,
                -0.0034793091,
                -0.009803017,
                -0.00091109646,
                0.009955396,
                0.00066229096,
                -0.008228439,
                0.0014301358,
                0.0035364511,
                -0.029104294,
                0.02275519,
                0.026539257,
                0.011110933,
                0.0053776912,
                0.0042538997,
                -0.002219012,
                -0.002598371,
                0.005238011,
                -0.015428323,
                -0.005501499,
                -0.021790126,
                -0.016266406,
                0.025142454,
                -0.008526847,
                -0.0032364558,
                -0.008082409,
                0.0018983822,
                -0.009244296,
                0.00045237367,
                0.043631043,
                -0.008412563,
                -0.011339501,
                -0.025040867,
                0.0041078706,
                -0.011034743,
                0.020533003,
                0.0147680165,
                -0.018971123,
                0.007415754,
                -0.012012505,
                0.030374115,
                0.012152186,
                0.032278847,
                0.0035777204,
                -0.009091917,
                -0.0074728956,
                -0.00652053,
                0.02502817,
                -0.012374404,
                0.009396674,
                -0.0009761748,
                0.012425197,
                -0.013295025,
                -0.0015912442,
                -0.012768049,
                -0.0037174006,
                0.0071744877,
                -0.010507768,
                0.009968094,
                -0.0043427874,
                -0.0025777363,
                -0.0041713617,
                0.010812525,
                -0.0255107,
                -0.02417739,
                -0.04012634,
                -0.015199755,
                0.012825191,
                -0.013244231,
                -0.028520176,
                -0.02360597,
                -0.0048792865,
                -0.008076061,
                -0.016152121,
                0.051351555,
                0.011783938,
              
                -0.025421813,
                0.030424908,
                -0.009079219,
                0.046627823,
                0.0008007808,
                0.0015856888,
                0.012806144,
                -0.000780543,
                0.0032793123,
                -0.10768081,
                -0.022996455,
                0.020609193,
                0.020507608,
                -0.0062633916,
                -0.0052475347,
                0.009199852,
                0.013472799,
                -0.01438707,
                0.0035618476,
                -0.011206169,
                -0.018018758,
                -0.0152251525,
                -0.013739462,
                0.023644064,
                
            ]

# Check the size of the vector data
vector_size = len(vector_data_ada_002)

# Determine the appropriate shape
features_per_data_point = 3  # Assuming each data point has 3 features
rows = vector_size // features_per_data_point

# Check if the size is divisible by 3
if vector_size % features_per_data_point != 0:
    raise ValueError(f"Vector data size ({vector_size}) is not divisible by {features_per_data_point}")

# Reshape the vector data into a 2D array
data_points = np.array(vector_data_ada_002).reshape(rows, features_per_data_point)

# Standardize the data
scaler = StandardScaler()
data_points_standardized = scaler.fit_transform(data_points)

# Apply K-means clustering
k = 4  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(data_points_standardized)

# Use PCA for dimensionality reduction (optional)
pca = PCA(n_components=2)
data_points_pca = pca.fit_transform(data_points_standardized)

# Display the clusters and data
print("Data Points Shape:")
print(data_points.shape)
print("\nClusters:")
print(clusters)
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Plot the clustered data
plt.scatter(data_points_pca[:, 0], data_points_pca[:, 1], c=clusters, cmap='viridis')
plt.title('K-means Clustering (ada-002)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()