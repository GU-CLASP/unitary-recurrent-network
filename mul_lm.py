
import tensorflow as tf
def mkModel():
  
  #[12,2500]
  var12345=tf.random.uniform([12,2500], minval=-5.0e-2, maxval=5.0e-2, dtype=tf.float32) # 0
  var12346=tf.Variable(name="embs", trainable=True, initial_value=var12345)
  #[50,12]
  var12347=tf.random.uniform(
             [50,12], minval=-0.3110855, maxval=0.3110855, dtype=tf.float32) # 4
  var12348=tf.Variable(name="projection_w", trainable=True, initial_value=var12347)
  #[12]
  var12349=tf.random.truncated_normal([12], stddev=0.1, dtype=tf.float32) # 5
  var12350=tf.Variable(name="projection_bias", trainable=True, initial_value=var12349)
  return {"batch_size":512
         ,"parameters":[var12346,var12348,var12350]
         ,"paramsdict":{"embs":var12346,"projection_w":var12348,"projection_bias":var12350}}
@tf.function
def runModel_fn(training_placeholder, embs, projection_w, projection_bias, x, y, weights):
  
  #[512,21]
  var12351=y
  #[]
  var12352=training_placeholder
  #[512,50]
  var12353=tf.random.uniform([512,50], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[512,50]
  var12354=tf.floor(var12353)
  #[]
  var12355=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[50]
  var12356=tf.broadcast_to(tf.reshape(var12355, [1]), [50])
  #[50]
  var12357=tf.reshape(var12356, [50])
  #[512,50]
  var12358=tf.broadcast_to(tf.reshape(var12357, [1,50]), [512,50])
  #[512,50]
  var12359=tf.divide(var12354, var12358)
  #[]
  var12360=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[50]
  var12361=tf.broadcast_to(tf.reshape(var12360, [1]), [50])
  #[50]
  var12362=tf.reshape(var12361, [50])
  #[512,50]
  var12363=tf.broadcast_to(tf.reshape(var12362, [1,50]), [512,50])
  #[512,50]
  var12364=tf.cond(var12352, true_fn=lambda: var12359, false_fn=lambda: var12363)
  #[512,50]
  var12365=tf.random.uniform([512,50], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[512,50]
  var12366=tf.floor(var12365)
  #[512,50]
  var12367=tf.divide(var12366, var12358)
  #[512,50]
  var12368=tf.cond(var12352, true_fn=lambda: var12367, false_fn=lambda: var12363)
  #[]
  var12369=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var12370=tf.broadcast_to(tf.reshape(var12369, [1]), [1])
  #[]
  var12371=tf.reshape(var12370, [])
  #[50]
  var12372=tf.one_hot(var12371, axis=0, dtype=tf.float32, depth=50)
  #[512,50]
  var12373=tf.broadcast_to(tf.reshape(var12372, [1,50]), [512,50])
  #[512,50]
  var12374=tf.multiply(var12368, var12373)
  #[512,1,50]
  var12375=tf.reshape(var12374, [512,1,50])
  #[512,2500]
  var12376=tf.random.uniform([512,2500], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[512,2500]
  var12377=tf.floor(var12376)
  #[2500]
  var12378=tf.broadcast_to(tf.reshape(var12355, [1]), [2500])
  #[2500]
  var12379=tf.reshape(var12378, [2500])
  #[512,2500]
  var12380=tf.broadcast_to(tf.reshape(var12379, [1,2500]), [512,2500])
  #[512,2500]
  var12381=tf.divide(var12377, var12380)
  #[2500]
  var12382=tf.broadcast_to(tf.reshape(var12360, [1]), [2500])
  #[2500]
  var12383=tf.reshape(var12382, [2500])
  #[512,2500]
  var12384=tf.broadcast_to(tf.reshape(var12383, [1,2500]), [512,2500])
  #[512,2500]
  var12385=tf.cond(var12352, true_fn=lambda: var12381, false_fn=lambda: var12384)
  #[12,2500]
  var12386=embs
  #[512,21]
  var12387=x
  #[512,1]
  var12388=var12387[:,0:1]
  #[512]
  var12389=tf.reshape(var12388, [512])
  #[512,2500]
  var12390=tf.gather(params=var12386, indices=var12389, batch_dims=0, axis=0)
  #[512,2500]
  var12391=tf.multiply(var12385, var12390)
  #[512,50,50]
  var12392=tf.reshape(var12391, [512,50,50])
  #[512,1,50]
  var12393=tf.matmul(var12375, var12392)
  #[512,50]
  var12394=tf.reshape(var12393, [512,50])
  #[512,50]
  var12395=tf.multiply(var12364, var12394)
  #[512,50]
  var12396=tf.reshape(var12395, [512,50])
  #[50,12]
  var12397=projection_w
  #[512,12]
  var12398=tf.matmul(var12396, var12397)
  #[512,12]
  var12399=tf.reshape(var12398, [512,12])
  #[12]
  var12400=projection_bias
  #[512,12]
  var12401=tf.broadcast_to(tf.reshape(var12400, [1,12]), [512,12])
  #[512,12]
  var12402=tf.add(var12399, var12401)
  #[512,1,12]
  var12403=tf.reshape(var12402, [512,1,12])
  #[512,50]
  var12404=tf.multiply(var12368, var12394)
  #[512,1,50]
  var12405=tf.reshape(var12404, [512,1,50])
  #[512,1]
  var12406=var12387[:,1:2]
  #[512]
  var12407=tf.reshape(var12406, [512])
  #[512,2500]
  var12408=tf.gather(params=var12386, indices=var12407, batch_dims=0, axis=0)
  #[512,2500]
  var12409=tf.multiply(var12385, var12408)
  #[512,50,50]
  var12410=tf.reshape(var12409, [512,50,50])
  #[512,1,50]
  var12411=tf.matmul(var12405, var12410)
  #[512,50]
  var12412=tf.reshape(var12411, [512,50])
  #[512,50]
  var12413=tf.multiply(var12364, var12412)
  #[512,50]
  var12414=tf.reshape(var12413, [512,50])
  #[512,12]
  var12415=tf.matmul(var12414, var12397)
  #[512,12]
  var12416=tf.reshape(var12415, [512,12])
  #[512,12]
  var12417=tf.add(var12416, var12401)
  #[512,1,12]
  var12418=tf.reshape(var12417, [512,1,12])
  #[512,50]
  var12419=tf.multiply(var12368, var12412)
  #[512,1,50]
  var12420=tf.reshape(var12419, [512,1,50])
  #[512,1]
  var12421=var12387[:,2:3]
  #[512]
  var12422=tf.reshape(var12421, [512])
  #[512,2500]
  var12423=tf.gather(params=var12386, indices=var12422, batch_dims=0, axis=0)
  #[512,2500]
  var12424=tf.multiply(var12385, var12423)
  #[512,50,50]
  var12425=tf.reshape(var12424, [512,50,50])
  #[512,1,50]
  var12426=tf.matmul(var12420, var12425)
  #[512,50]
  var12427=tf.reshape(var12426, [512,50])
  #[512,50]
  var12428=tf.multiply(var12364, var12427)
  #[512,50]
  var12429=tf.reshape(var12428, [512,50])
  #[512,12]
  var12430=tf.matmul(var12429, var12397)
  #[512,12]
  var12431=tf.reshape(var12430, [512,12])
  #[512,12]
  var12432=tf.add(var12431, var12401)
  #[512,1,12]
  var12433=tf.reshape(var12432, [512,1,12])
  #[512,50]
  var12434=tf.multiply(var12368, var12427)
  #[512,1,50]
  var12435=tf.reshape(var12434, [512,1,50])
  #[512,1]
  var12436=var12387[:,3:4]
  #[512]
  var12437=tf.reshape(var12436, [512])
  #[512,2500]
  var12438=tf.gather(params=var12386, indices=var12437, batch_dims=0, axis=0)
  #[512,2500]
  var12439=tf.multiply(var12385, var12438)
  #[512,50,50]
  var12440=tf.reshape(var12439, [512,50,50])
  #[512,1,50]
  var12441=tf.matmul(var12435, var12440)
  #[512,50]
  var12442=tf.reshape(var12441, [512,50])
  #[512,50]
  var12443=tf.multiply(var12364, var12442)
  #[512,50]
  var12444=tf.reshape(var12443, [512,50])
  #[512,12]
  var12445=tf.matmul(var12444, var12397)
  #[512,12]
  var12446=tf.reshape(var12445, [512,12])
  #[512,12]
  var12447=tf.add(var12446, var12401)
  #[512,1,12]
  var12448=tf.reshape(var12447, [512,1,12])
  #[512,50]
  var12449=tf.multiply(var12368, var12442)
  #[512,1,50]
  var12450=tf.reshape(var12449, [512,1,50])
  #[512,1]
  var12451=var12387[:,4:5]
  #[512]
  var12452=tf.reshape(var12451, [512])
  #[512,2500]
  var12453=tf.gather(params=var12386, indices=var12452, batch_dims=0, axis=0)
  #[512,2500]
  var12454=tf.multiply(var12385, var12453)
  #[512,50,50]
  var12455=tf.reshape(var12454, [512,50,50])
  #[512,1,50]
  var12456=tf.matmul(var12450, var12455)
  #[512,50]
  var12457=tf.reshape(var12456, [512,50])
  #[512,50]
  var12458=tf.multiply(var12364, var12457)
  #[512,50]
  var12459=tf.reshape(var12458, [512,50])
  #[512,12]
  var12460=tf.matmul(var12459, var12397)
  #[512,12]
  var12461=tf.reshape(var12460, [512,12])
  #[512,12]
  var12462=tf.add(var12461, var12401)
  #[512,1,12]
  var12463=tf.reshape(var12462, [512,1,12])
  #[512,50]
  var12464=tf.multiply(var12368, var12457)
  #[512,1,50]
  var12465=tf.reshape(var12464, [512,1,50])
  #[512,1]
  var12466=var12387[:,5:6]
  #[512]
  var12467=tf.reshape(var12466, [512])
  #[512,2500]
  var12468=tf.gather(params=var12386, indices=var12467, batch_dims=0, axis=0)
  #[512,2500]
  var12469=tf.multiply(var12385, var12468)
  #[512,50,50]
  var12470=tf.reshape(var12469, [512,50,50])
  #[512,1,50]
  var12471=tf.matmul(var12465, var12470)
  #[512,50]
  var12472=tf.reshape(var12471, [512,50])
  #[512,50]
  var12473=tf.multiply(var12364, var12472)
  #[512,50]
  var12474=tf.reshape(var12473, [512,50])
  #[512,12]
  var12475=tf.matmul(var12474, var12397)
  #[512,12]
  var12476=tf.reshape(var12475, [512,12])
  #[512,12]
  var12477=tf.add(var12476, var12401)
  #[512,1,12]
  var12478=tf.reshape(var12477, [512,1,12])
  #[512,50]
  var12479=tf.multiply(var12368, var12472)
  #[512,1,50]
  var12480=tf.reshape(var12479, [512,1,50])
  #[512,1]
  var12481=var12387[:,6:7]
  #[512]
  var12482=tf.reshape(var12481, [512])
  #[512,2500]
  var12483=tf.gather(params=var12386, indices=var12482, batch_dims=0, axis=0)
  #[512,2500]
  var12484=tf.multiply(var12385, var12483)
  #[512,50,50]
  var12485=tf.reshape(var12484, [512,50,50])
  #[512,1,50]
  var12486=tf.matmul(var12480, var12485)
  #[512,50]
  var12487=tf.reshape(var12486, [512,50])
  #[512,50]
  var12488=tf.multiply(var12364, var12487)
  #[512,50]
  var12489=tf.reshape(var12488, [512,50])
  #[512,12]
  var12490=tf.matmul(var12489, var12397)
  #[512,12]
  var12491=tf.reshape(var12490, [512,12])
  #[512,12]
  var12492=tf.add(var12491, var12401)
  #[512,1,12]
  var12493=tf.reshape(var12492, [512,1,12])
  #[512,50]
  var12494=tf.multiply(var12368, var12487)
  #[512,1,50]
  var12495=tf.reshape(var12494, [512,1,50])
  #[512,1]
  var12496=var12387[:,7:8]
  #[512]
  var12497=tf.reshape(var12496, [512])
  #[512,2500]
  var12498=tf.gather(params=var12386, indices=var12497, batch_dims=0, axis=0)
  #[512,2500]
  var12499=tf.multiply(var12385, var12498)
  #[512,50,50]
  var12500=tf.reshape(var12499, [512,50,50])
  #[512,1,50]
  var12501=tf.matmul(var12495, var12500)
  #[512,50]
  var12502=tf.reshape(var12501, [512,50])
  #[512,50]
  var12503=tf.multiply(var12364, var12502)
  #[512,50]
  var12504=tf.reshape(var12503, [512,50])
  #[512,12]
  var12505=tf.matmul(var12504, var12397)
  #[512,12]
  var12506=tf.reshape(var12505, [512,12])
  #[512,12]
  var12507=tf.add(var12506, var12401)
  #[512,1,12]
  var12508=tf.reshape(var12507, [512,1,12])
  #[512,50]
  var12509=tf.multiply(var12368, var12502)
  #[512,1,50]
  var12510=tf.reshape(var12509, [512,1,50])
  #[512,1]
  var12511=var12387[:,8:9]
  #[512]
  var12512=tf.reshape(var12511, [512])
  #[512,2500]
  var12513=tf.gather(params=var12386, indices=var12512, batch_dims=0, axis=0)
  #[512,2500]
  var12514=tf.multiply(var12385, var12513)
  #[512,50,50]
  var12515=tf.reshape(var12514, [512,50,50])
  #[512,1,50]
  var12516=tf.matmul(var12510, var12515)
  #[512,50]
  var12517=tf.reshape(var12516, [512,50])
  #[512,50]
  var12518=tf.multiply(var12364, var12517)
  #[512,50]
  var12519=tf.reshape(var12518, [512,50])
  #[512,12]
  var12520=tf.matmul(var12519, var12397)
  #[512,12]
  var12521=tf.reshape(var12520, [512,12])
  #[512,12]
  var12522=tf.add(var12521, var12401)
  #[512,1,12]
  var12523=tf.reshape(var12522, [512,1,12])
  #[512,50]
  var12524=tf.multiply(var12368, var12517)
  #[512,1,50]
  var12525=tf.reshape(var12524, [512,1,50])
  #[512,1]
  var12526=var12387[:,9:10]
  #[512]
  var12527=tf.reshape(var12526, [512])
  #[512,2500]
  var12528=tf.gather(params=var12386, indices=var12527, batch_dims=0, axis=0)
  #[512,2500]
  var12529=tf.multiply(var12385, var12528)
  #[512,50,50]
  var12530=tf.reshape(var12529, [512,50,50])
  #[512,1,50]
  var12531=tf.matmul(var12525, var12530)
  #[512,50]
  var12532=tf.reshape(var12531, [512,50])
  #[512,50]
  var12533=tf.multiply(var12364, var12532)
  #[512,50]
  var12534=tf.reshape(var12533, [512,50])
  #[512,12]
  var12535=tf.matmul(var12534, var12397)
  #[512,12]
  var12536=tf.reshape(var12535, [512,12])
  #[512,12]
  var12537=tf.add(var12536, var12401)
  #[512,1,12]
  var12538=tf.reshape(var12537, [512,1,12])
  #[512,50]
  var12539=tf.multiply(var12368, var12532)
  #[512,1,50]
  var12540=tf.reshape(var12539, [512,1,50])
  #[512,1]
  var12541=var12387[:,10:11]
  #[512]
  var12542=tf.reshape(var12541, [512])
  #[512,2500]
  var12543=tf.gather(params=var12386, indices=var12542, batch_dims=0, axis=0)
  #[512,2500]
  var12544=tf.multiply(var12385, var12543)
  #[512,50,50]
  var12545=tf.reshape(var12544, [512,50,50])
  #[512,1,50]
  var12546=tf.matmul(var12540, var12545)
  #[512,50]
  var12547=tf.reshape(var12546, [512,50])
  #[512,50]
  var12548=tf.multiply(var12364, var12547)
  #[512,50]
  var12549=tf.reshape(var12548, [512,50])
  #[512,12]
  var12550=tf.matmul(var12549, var12397)
  #[512,12]
  var12551=tf.reshape(var12550, [512,12])
  #[512,12]
  var12552=tf.add(var12551, var12401)
  #[512,1,12]
  var12553=tf.reshape(var12552, [512,1,12])
  #[512,50]
  var12554=tf.multiply(var12368, var12547)
  #[512,1,50]
  var12555=tf.reshape(var12554, [512,1,50])
  #[512,1]
  var12556=var12387[:,11:12]
  #[512]
  var12557=tf.reshape(var12556, [512])
  #[512,2500]
  var12558=tf.gather(params=var12386, indices=var12557, batch_dims=0, axis=0)
  #[512,2500]
  var12559=tf.multiply(var12385, var12558)
  #[512,50,50]
  var12560=tf.reshape(var12559, [512,50,50])
  #[512,1,50]
  var12561=tf.matmul(var12555, var12560)
  #[512,50]
  var12562=tf.reshape(var12561, [512,50])
  #[512,50]
  var12563=tf.multiply(var12364, var12562)
  #[512,50]
  var12564=tf.reshape(var12563, [512,50])
  #[512,12]
  var12565=tf.matmul(var12564, var12397)
  #[512,12]
  var12566=tf.reshape(var12565, [512,12])
  #[512,12]
  var12567=tf.add(var12566, var12401)
  #[512,1,12]
  var12568=tf.reshape(var12567, [512,1,12])
  #[512,50]
  var12569=tf.multiply(var12368, var12562)
  #[512,1,50]
  var12570=tf.reshape(var12569, [512,1,50])
  #[512,1]
  var12571=var12387[:,12:13]
  #[512]
  var12572=tf.reshape(var12571, [512])
  #[512,2500]
  var12573=tf.gather(params=var12386, indices=var12572, batch_dims=0, axis=0)
  #[512,2500]
  var12574=tf.multiply(var12385, var12573)
  #[512,50,50]
  var12575=tf.reshape(var12574, [512,50,50])
  #[512,1,50]
  var12576=tf.matmul(var12570, var12575)
  #[512,50]
  var12577=tf.reshape(var12576, [512,50])
  #[512,50]
  var12578=tf.multiply(var12364, var12577)
  #[512,50]
  var12579=tf.reshape(var12578, [512,50])
  #[512,12]
  var12580=tf.matmul(var12579, var12397)
  #[512,12]
  var12581=tf.reshape(var12580, [512,12])
  #[512,12]
  var12582=tf.add(var12581, var12401)
  #[512,1,12]
  var12583=tf.reshape(var12582, [512,1,12])
  #[512,50]
  var12584=tf.multiply(var12368, var12577)
  #[512,1,50]
  var12585=tf.reshape(var12584, [512,1,50])
  #[512,1]
  var12586=var12387[:,13:14]
  #[512]
  var12587=tf.reshape(var12586, [512])
  #[512,2500]
  var12588=tf.gather(params=var12386, indices=var12587, batch_dims=0, axis=0)
  #[512,2500]
  var12589=tf.multiply(var12385, var12588)
  #[512,50,50]
  var12590=tf.reshape(var12589, [512,50,50])
  #[512,1,50]
  var12591=tf.matmul(var12585, var12590)
  #[512,50]
  var12592=tf.reshape(var12591, [512,50])
  #[512,50]
  var12593=tf.multiply(var12364, var12592)
  #[512,50]
  var12594=tf.reshape(var12593, [512,50])
  #[512,12]
  var12595=tf.matmul(var12594, var12397)
  #[512,12]
  var12596=tf.reshape(var12595, [512,12])
  #[512,12]
  var12597=tf.add(var12596, var12401)
  #[512,1,12]
  var12598=tf.reshape(var12597, [512,1,12])
  #[512,50]
  var12599=tf.multiply(var12368, var12592)
  #[512,1,50]
  var12600=tf.reshape(var12599, [512,1,50])
  #[512,1]
  var12601=var12387[:,14:15]
  #[512]
  var12602=tf.reshape(var12601, [512])
  #[512,2500]
  var12603=tf.gather(params=var12386, indices=var12602, batch_dims=0, axis=0)
  #[512,2500]
  var12604=tf.multiply(var12385, var12603)
  #[512,50,50]
  var12605=tf.reshape(var12604, [512,50,50])
  #[512,1,50]
  var12606=tf.matmul(var12600, var12605)
  #[512,50]
  var12607=tf.reshape(var12606, [512,50])
  #[512,50]
  var12608=tf.multiply(var12364, var12607)
  #[512,50]
  var12609=tf.reshape(var12608, [512,50])
  #[512,12]
  var12610=tf.matmul(var12609, var12397)
  #[512,12]
  var12611=tf.reshape(var12610, [512,12])
  #[512,12]
  var12612=tf.add(var12611, var12401)
  #[512,1,12]
  var12613=tf.reshape(var12612, [512,1,12])
  #[512,50]
  var12614=tf.multiply(var12368, var12607)
  #[512,1,50]
  var12615=tf.reshape(var12614, [512,1,50])
  #[512,1]
  var12616=var12387[:,15:16]
  #[512]
  var12617=tf.reshape(var12616, [512])
  #[512,2500]
  var12618=tf.gather(params=var12386, indices=var12617, batch_dims=0, axis=0)
  #[512,2500]
  var12619=tf.multiply(var12385, var12618)
  #[512,50,50]
  var12620=tf.reshape(var12619, [512,50,50])
  #[512,1,50]
  var12621=tf.matmul(var12615, var12620)
  #[512,50]
  var12622=tf.reshape(var12621, [512,50])
  #[512,50]
  var12623=tf.multiply(var12364, var12622)
  #[512,50]
  var12624=tf.reshape(var12623, [512,50])
  #[512,12]
  var12625=tf.matmul(var12624, var12397)
  #[512,12]
  var12626=tf.reshape(var12625, [512,12])
  #[512,12]
  var12627=tf.add(var12626, var12401)
  #[512,1,12]
  var12628=tf.reshape(var12627, [512,1,12])
  #[512,50]
  var12629=tf.multiply(var12368, var12622)
  #[512,1,50]
  var12630=tf.reshape(var12629, [512,1,50])
  #[512,1]
  var12631=var12387[:,16:17]
  #[512]
  var12632=tf.reshape(var12631, [512])
  #[512,2500]
  var12633=tf.gather(params=var12386, indices=var12632, batch_dims=0, axis=0)
  #[512,2500]
  var12634=tf.multiply(var12385, var12633)
  #[512,50,50]
  var12635=tf.reshape(var12634, [512,50,50])
  #[512,1,50]
  var12636=tf.matmul(var12630, var12635)
  #[512,50]
  var12637=tf.reshape(var12636, [512,50])
  #[512,50]
  var12638=tf.multiply(var12364, var12637)
  #[512,50]
  var12639=tf.reshape(var12638, [512,50])
  #[512,12]
  var12640=tf.matmul(var12639, var12397)
  #[512,12]
  var12641=tf.reshape(var12640, [512,12])
  #[512,12]
  var12642=tf.add(var12641, var12401)
  #[512,1,12]
  var12643=tf.reshape(var12642, [512,1,12])
  #[512,50]
  var12644=tf.multiply(var12368, var12637)
  #[512,1,50]
  var12645=tf.reshape(var12644, [512,1,50])
  #[512,1]
  var12646=var12387[:,17:18]
  #[512]
  var12647=tf.reshape(var12646, [512])
  #[512,2500]
  var12648=tf.gather(params=var12386, indices=var12647, batch_dims=0, axis=0)
  #[512,2500]
  var12649=tf.multiply(var12385, var12648)
  #[512,50,50]
  var12650=tf.reshape(var12649, [512,50,50])
  #[512,1,50]
  var12651=tf.matmul(var12645, var12650)
  #[512,50]
  var12652=tf.reshape(var12651, [512,50])
  #[512,50]
  var12653=tf.multiply(var12364, var12652)
  #[512,50]
  var12654=tf.reshape(var12653, [512,50])
  #[512,12]
  var12655=tf.matmul(var12654, var12397)
  #[512,12]
  var12656=tf.reshape(var12655, [512,12])
  #[512,12]
  var12657=tf.add(var12656, var12401)
  #[512,1,12]
  var12658=tf.reshape(var12657, [512,1,12])
  #[512,50]
  var12659=tf.multiply(var12368, var12652)
  #[512,1,50]
  var12660=tf.reshape(var12659, [512,1,50])
  #[512,1]
  var12661=var12387[:,18:19]
  #[512]
  var12662=tf.reshape(var12661, [512])
  #[512,2500]
  var12663=tf.gather(params=var12386, indices=var12662, batch_dims=0, axis=0)
  #[512,2500]
  var12664=tf.multiply(var12385, var12663)
  #[512,50,50]
  var12665=tf.reshape(var12664, [512,50,50])
  #[512,1,50]
  var12666=tf.matmul(var12660, var12665)
  #[512,50]
  var12667=tf.reshape(var12666, [512,50])
  #[512,50]
  var12668=tf.multiply(var12364, var12667)
  #[512,50]
  var12669=tf.reshape(var12668, [512,50])
  #[512,12]
  var12670=tf.matmul(var12669, var12397)
  #[512,12]
  var12671=tf.reshape(var12670, [512,12])
  #[512,12]
  var12672=tf.add(var12671, var12401)
  #[512,1,12]
  var12673=tf.reshape(var12672, [512,1,12])
  #[512,50]
  var12674=tf.multiply(var12368, var12667)
  #[512,1,50]
  var12675=tf.reshape(var12674, [512,1,50])
  #[512,1]
  var12676=var12387[:,19:20]
  #[512]
  var12677=tf.reshape(var12676, [512])
  #[512,2500]
  var12678=tf.gather(params=var12386, indices=var12677, batch_dims=0, axis=0)
  #[512,2500]
  var12679=tf.multiply(var12385, var12678)
  #[512,50,50]
  var12680=tf.reshape(var12679, [512,50,50])
  #[512,1,50]
  var12681=tf.matmul(var12675, var12680)
  #[512,50]
  var12682=tf.reshape(var12681, [512,50])
  #[512,50]
  var12683=tf.multiply(var12364, var12682)
  #[512,50]
  var12684=tf.reshape(var12683, [512,50])
  #[512,12]
  var12685=tf.matmul(var12684, var12397)
  #[512,12]
  var12686=tf.reshape(var12685, [512,12])
  #[512,12]
  var12687=tf.add(var12686, var12401)
  #[512,1,12]
  var12688=tf.reshape(var12687, [512,1,12])
  #[512,50]
  var12689=tf.multiply(var12368, var12682)
  #[512,1,50]
  var12690=tf.reshape(var12689, [512,1,50])
  #[512,1]
  var12691=var12387[:,20:21]
  #[512]
  var12692=tf.reshape(var12691, [512])
  #[512,2500]
  var12693=tf.gather(params=var12386, indices=var12692, batch_dims=0, axis=0)
  #[512,2500]
  var12694=tf.multiply(var12385, var12693)
  #[512,50,50]
  var12695=tf.reshape(var12694, [512,50,50])
  #[512,1,50]
  var12696=tf.matmul(var12690, var12695)
  #[512,50]
  var12697=tf.reshape(var12696, [512,50])
  #[512,50]
  var12698=tf.multiply(var12364, var12697)
  #[512,50]
  var12699=tf.reshape(var12698, [512,50])
  #[512,12]
  var12700=tf.matmul(var12699, var12397)
  #[512,12]
  var12701=tf.reshape(var12700, [512,12])
  #[512,12]
  var12702=tf.add(var12701, var12401)
  #[512,1,12]
  var12703=tf.reshape(var12702, [512,1,12])
  #[512,21,12]
  var12704=tf.concat([var12403
                     ,var12418
                     ,var12433
                     ,var12448
                     ,var12463
                     ,var12478
                     ,var12493
                     ,var12508
                     ,var12523
                     ,var12538
                     ,var12553
                     ,var12568
                     ,var12583
                     ,var12598
                     ,var12613
                     ,var12628
                     ,var12643
                     ,var12658
                     ,var12673
                     ,var12688
                     ,var12703],
                     axis=1)
  #[512,21]
  var12705=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=var12351, logits=var12704)
  #[512,21]
  var12706=weights
  #[512,21]
  var12707=tf.multiply(var12705, var12706)
  #[512,21]
  var12708=tf.reshape(var12707, [512,21])
  #[512]
  var12709=tf.reduce_sum(var12708, axis=1)
  #[512,21]
  var12710=tf.reshape(var12706, [512,21])
  #[512]
  var12711=tf.reduce_sum(var12710, axis=1)
  #[512]
  var12712=tf.divide(var12709, var12711)
  #[512]
  var12713=tf.cast(var12712, tf.float32)
  #[512]
  var12714=tf.reshape(var12713, [512])
  #[]
  var12715=tf.reduce_mean(var12714, axis=0)
  #[]
  var12716=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var12717=tf.broadcast_to(tf.reshape(var12716, [1]), [1])
  #[]
  var12718=tf.reshape(var12717, [])
  #[]
  var12719=tf.add(var12715, var12718)
  #[512,21]
  var12720=tf.argmax(var12704, axis=2, output_type=tf.int32)
  #[512,21]
  var12721=tf.equal(var12720, var12351)
  #[512,21]
  var12722=tf.cast(var12721, tf.float32)
  #[512,21]
  var12723=tf.multiply(var12722, var12706)
  #[512,21]
  var12724=tf.reshape(var12723, [512,21])
  #[512]
  var12725=tf.reduce_sum(var12724, axis=1)
  #[512]
  var12726=tf.divide(var12725, var12711)
  #[512]
  var12727=tf.cast(var12726, tf.float32)
  #[512]
  var12728=tf.reshape(var12727, [512])
  #[]
  var12729=tf.reduce_mean(var12728, axis=0)
  #[10752,12]
  var12730=tf.reshape(var12704, [10752,12])
  #[10752,12]
  var12731=tf.nn.softmax(var12730, axis=1)
  #[512,21,12]
  var12732=tf.reshape(var12731, [512,21,12])
  return {"loss":var12719,"accuracy":var12729,"y_":var12732}
runModel = {"function":runModel_fn
           ,"batched":True
           ,"placeholders":{"x":{"shape":[512,21],"dtype":tf.int32}
                           ,"y":{"shape":[512,21],"dtype":tf.int32}
                           ,"weights":{"shape":[512,21],"dtype":tf.float32}}}
@tf.function
def probeStates_fn(training_placeholder, embs, projection_w, projection_bias, x):
  
  #[]
  var12733=training_placeholder
  #[50]
  var12734=tf.random.uniform([50], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[50]
  var12735=tf.floor(var12734)
  #[]
  var12736=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[50]
  var12737=tf.broadcast_to(tf.reshape(var12736, [1]), [50])
  #[50]
  var12738=tf.reshape(var12737, [50])
  #[50]
  var12739=tf.divide(var12735, var12738)
  #[]
  var12740=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[50]
  var12741=tf.broadcast_to(tf.reshape(var12740, [1]), [50])
  #[50]
  var12742=tf.reshape(var12741, [50])
  #[50]
  var12743=tf.cond(var12733, true_fn=lambda: var12739, false_fn=lambda: var12742)
  #[50]
  var12744=tf.random.uniform([50], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[50]
  var12745=tf.floor(var12744)
  #[50]
  var12746=tf.divide(var12745, var12738)
  #[50]
  var12747=tf.cond(var12733, true_fn=lambda: var12746, false_fn=lambda: var12742)
  #[]
  var12748=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var12749=tf.broadcast_to(tf.reshape(var12748, [1]), [1])
  #[]
  var12750=tf.reshape(var12749, [])
  #[50]
  var12751=tf.one_hot(var12750, axis=0, dtype=tf.float32, depth=50)
  #[50]
  var12752=tf.multiply(var12747, var12751)
  #[1,50]
  var12753=tf.reshape(var12752, [1,50])
  #[2500]
  var12754=tf.random.uniform([2500], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[2500]
  var12755=tf.floor(var12754)
  #[2500]
  var12756=tf.broadcast_to(tf.reshape(var12736, [1]), [2500])
  #[2500]
  var12757=tf.reshape(var12756, [2500])
  #[2500]
  var12758=tf.divide(var12755, var12757)
  #[2500]
  var12759=tf.broadcast_to(tf.reshape(var12740, [1]), [2500])
  #[2500]
  var12760=tf.reshape(var12759, [2500])
  #[2500]
  var12761=tf.cond(var12733, true_fn=lambda: var12758, false_fn=lambda: var12760)
  #[12,2500]
  var12762=embs
  #[21]
  var12763=x
  #[1]
  var12764=var12763[0:1]
  #[]
  var12765=tf.reshape(var12764, [])
  #[2500]
  var12766=tf.gather(params=var12762, indices=var12765, batch_dims=0, axis=0)
  #[2500]
  var12767=tf.multiply(var12761, var12766)
  #[50,50]
  var12768=tf.reshape(var12767, [50,50])
  #[1,50]
  var12769=tf.matmul(var12753, var12768)
  #[50]
  var12770=tf.reshape(var12769, [50])
  #[50]
  var12771=tf.multiply(var12743, var12770)
  #[1,50]
  var12772=tf.reshape(var12771, [1,50])
  #[50]
  var12773=tf.multiply(var12747, var12770)
  #[1,50]
  var12774=tf.reshape(var12773, [1,50])
  #[1]
  var12775=var12763[1:2]
  #[]
  var12776=tf.reshape(var12775, [])
  #[2500]
  var12777=tf.gather(params=var12762, indices=var12776, batch_dims=0, axis=0)
  #[2500]
  var12778=tf.multiply(var12761, var12777)
  #[50,50]
  var12779=tf.reshape(var12778, [50,50])
  #[1,50]
  var12780=tf.matmul(var12774, var12779)
  #[50]
  var12781=tf.reshape(var12780, [50])
  #[50]
  var12782=tf.multiply(var12743, var12781)
  #[1,50]
  var12783=tf.reshape(var12782, [1,50])
  #[50]
  var12784=tf.multiply(var12747, var12781)
  #[1,50]
  var12785=tf.reshape(var12784, [1,50])
  #[1]
  var12786=var12763[2:3]
  #[]
  var12787=tf.reshape(var12786, [])
  #[2500]
  var12788=tf.gather(params=var12762, indices=var12787, batch_dims=0, axis=0)
  #[2500]
  var12789=tf.multiply(var12761, var12788)
  #[50,50]
  var12790=tf.reshape(var12789, [50,50])
  #[1,50]
  var12791=tf.matmul(var12785, var12790)
  #[50]
  var12792=tf.reshape(var12791, [50])
  #[50]
  var12793=tf.multiply(var12743, var12792)
  #[1,50]
  var12794=tf.reshape(var12793, [1,50])
  #[50]
  var12795=tf.multiply(var12747, var12792)
  #[1,50]
  var12796=tf.reshape(var12795, [1,50])
  #[1]
  var12797=var12763[3:4]
  #[]
  var12798=tf.reshape(var12797, [])
  #[2500]
  var12799=tf.gather(params=var12762, indices=var12798, batch_dims=0, axis=0)
  #[2500]
  var12800=tf.multiply(var12761, var12799)
  #[50,50]
  var12801=tf.reshape(var12800, [50,50])
  #[1,50]
  var12802=tf.matmul(var12796, var12801)
  #[50]
  var12803=tf.reshape(var12802, [50])
  #[50]
  var12804=tf.multiply(var12743, var12803)
  #[1,50]
  var12805=tf.reshape(var12804, [1,50])
  #[50]
  var12806=tf.multiply(var12747, var12803)
  #[1,50]
  var12807=tf.reshape(var12806, [1,50])
  #[1]
  var12808=var12763[4:5]
  #[]
  var12809=tf.reshape(var12808, [])
  #[2500]
  var12810=tf.gather(params=var12762, indices=var12809, batch_dims=0, axis=0)
  #[2500]
  var12811=tf.multiply(var12761, var12810)
  #[50,50]
  var12812=tf.reshape(var12811, [50,50])
  #[1,50]
  var12813=tf.matmul(var12807, var12812)
  #[50]
  var12814=tf.reshape(var12813, [50])
  #[50]
  var12815=tf.multiply(var12743, var12814)
  #[1,50]
  var12816=tf.reshape(var12815, [1,50])
  #[50]
  var12817=tf.multiply(var12747, var12814)
  #[1,50]
  var12818=tf.reshape(var12817, [1,50])
  #[1]
  var12819=var12763[5:6]
  #[]
  var12820=tf.reshape(var12819, [])
  #[2500]
  var12821=tf.gather(params=var12762, indices=var12820, batch_dims=0, axis=0)
  #[2500]
  var12822=tf.multiply(var12761, var12821)
  #[50,50]
  var12823=tf.reshape(var12822, [50,50])
  #[1,50]
  var12824=tf.matmul(var12818, var12823)
  #[50]
  var12825=tf.reshape(var12824, [50])
  #[50]
  var12826=tf.multiply(var12743, var12825)
  #[1,50]
  var12827=tf.reshape(var12826, [1,50])
  #[50]
  var12828=tf.multiply(var12747, var12825)
  #[1,50]
  var12829=tf.reshape(var12828, [1,50])
  #[1]
  var12830=var12763[6:7]
  #[]
  var12831=tf.reshape(var12830, [])
  #[2500]
  var12832=tf.gather(params=var12762, indices=var12831, batch_dims=0, axis=0)
  #[2500]
  var12833=tf.multiply(var12761, var12832)
  #[50,50]
  var12834=tf.reshape(var12833, [50,50])
  #[1,50]
  var12835=tf.matmul(var12829, var12834)
  #[50]
  var12836=tf.reshape(var12835, [50])
  #[50]
  var12837=tf.multiply(var12743, var12836)
  #[1,50]
  var12838=tf.reshape(var12837, [1,50])
  #[50]
  var12839=tf.multiply(var12747, var12836)
  #[1,50]
  var12840=tf.reshape(var12839, [1,50])
  #[1]
  var12841=var12763[7:8]
  #[]
  var12842=tf.reshape(var12841, [])
  #[2500]
  var12843=tf.gather(params=var12762, indices=var12842, batch_dims=0, axis=0)
  #[2500]
  var12844=tf.multiply(var12761, var12843)
  #[50,50]
  var12845=tf.reshape(var12844, [50,50])
  #[1,50]
  var12846=tf.matmul(var12840, var12845)
  #[50]
  var12847=tf.reshape(var12846, [50])
  #[50]
  var12848=tf.multiply(var12743, var12847)
  #[1,50]
  var12849=tf.reshape(var12848, [1,50])
  #[50]
  var12850=tf.multiply(var12747, var12847)
  #[1,50]
  var12851=tf.reshape(var12850, [1,50])
  #[1]
  var12852=var12763[8:9]
  #[]
  var12853=tf.reshape(var12852, [])
  #[2500]
  var12854=tf.gather(params=var12762, indices=var12853, batch_dims=0, axis=0)
  #[2500]
  var12855=tf.multiply(var12761, var12854)
  #[50,50]
  var12856=tf.reshape(var12855, [50,50])
  #[1,50]
  var12857=tf.matmul(var12851, var12856)
  #[50]
  var12858=tf.reshape(var12857, [50])
  #[50]
  var12859=tf.multiply(var12743, var12858)
  #[1,50]
  var12860=tf.reshape(var12859, [1,50])
  #[50]
  var12861=tf.multiply(var12747, var12858)
  #[1,50]
  var12862=tf.reshape(var12861, [1,50])
  #[1]
  var12863=var12763[9:10]
  #[]
  var12864=tf.reshape(var12863, [])
  #[2500]
  var12865=tf.gather(params=var12762, indices=var12864, batch_dims=0, axis=0)
  #[2500]
  var12866=tf.multiply(var12761, var12865)
  #[50,50]
  var12867=tf.reshape(var12866, [50,50])
  #[1,50]
  var12868=tf.matmul(var12862, var12867)
  #[50]
  var12869=tf.reshape(var12868, [50])
  #[50]
  var12870=tf.multiply(var12743, var12869)
  #[1,50]
  var12871=tf.reshape(var12870, [1,50])
  #[50]
  var12872=tf.multiply(var12747, var12869)
  #[1,50]
  var12873=tf.reshape(var12872, [1,50])
  #[1]
  var12874=var12763[10:11]
  #[]
  var12875=tf.reshape(var12874, [])
  #[2500]
  var12876=tf.gather(params=var12762, indices=var12875, batch_dims=0, axis=0)
  #[2500]
  var12877=tf.multiply(var12761, var12876)
  #[50,50]
  var12878=tf.reshape(var12877, [50,50])
  #[1,50]
  var12879=tf.matmul(var12873, var12878)
  #[50]
  var12880=tf.reshape(var12879, [50])
  #[50]
  var12881=tf.multiply(var12743, var12880)
  #[1,50]
  var12882=tf.reshape(var12881, [1,50])
  #[50]
  var12883=tf.multiply(var12747, var12880)
  #[1,50]
  var12884=tf.reshape(var12883, [1,50])
  #[1]
  var12885=var12763[11:12]
  #[]
  var12886=tf.reshape(var12885, [])
  #[2500]
  var12887=tf.gather(params=var12762, indices=var12886, batch_dims=0, axis=0)
  #[2500]
  var12888=tf.multiply(var12761, var12887)
  #[50,50]
  var12889=tf.reshape(var12888, [50,50])
  #[1,50]
  var12890=tf.matmul(var12884, var12889)
  #[50]
  var12891=tf.reshape(var12890, [50])
  #[50]
  var12892=tf.multiply(var12743, var12891)
  #[1,50]
  var12893=tf.reshape(var12892, [1,50])
  #[50]
  var12894=tf.multiply(var12747, var12891)
  #[1,50]
  var12895=tf.reshape(var12894, [1,50])
  #[1]
  var12896=var12763[12:13]
  #[]
  var12897=tf.reshape(var12896, [])
  #[2500]
  var12898=tf.gather(params=var12762, indices=var12897, batch_dims=0, axis=0)
  #[2500]
  var12899=tf.multiply(var12761, var12898)
  #[50,50]
  var12900=tf.reshape(var12899, [50,50])
  #[1,50]
  var12901=tf.matmul(var12895, var12900)
  #[50]
  var12902=tf.reshape(var12901, [50])
  #[50]
  var12903=tf.multiply(var12743, var12902)
  #[1,50]
  var12904=tf.reshape(var12903, [1,50])
  #[50]
  var12905=tf.multiply(var12747, var12902)
  #[1,50]
  var12906=tf.reshape(var12905, [1,50])
  #[1]
  var12907=var12763[13:14]
  #[]
  var12908=tf.reshape(var12907, [])
  #[2500]
  var12909=tf.gather(params=var12762, indices=var12908, batch_dims=0, axis=0)
  #[2500]
  var12910=tf.multiply(var12761, var12909)
  #[50,50]
  var12911=tf.reshape(var12910, [50,50])
  #[1,50]
  var12912=tf.matmul(var12906, var12911)
  #[50]
  var12913=tf.reshape(var12912, [50])
  #[50]
  var12914=tf.multiply(var12743, var12913)
  #[1,50]
  var12915=tf.reshape(var12914, [1,50])
  #[50]
  var12916=tf.multiply(var12747, var12913)
  #[1,50]
  var12917=tf.reshape(var12916, [1,50])
  #[1]
  var12918=var12763[14:15]
  #[]
  var12919=tf.reshape(var12918, [])
  #[2500]
  var12920=tf.gather(params=var12762, indices=var12919, batch_dims=0, axis=0)
  #[2500]
  var12921=tf.multiply(var12761, var12920)
  #[50,50]
  var12922=tf.reshape(var12921, [50,50])
  #[1,50]
  var12923=tf.matmul(var12917, var12922)
  #[50]
  var12924=tf.reshape(var12923, [50])
  #[50]
  var12925=tf.multiply(var12743, var12924)
  #[1,50]
  var12926=tf.reshape(var12925, [1,50])
  #[50]
  var12927=tf.multiply(var12747, var12924)
  #[1,50]
  var12928=tf.reshape(var12927, [1,50])
  #[1]
  var12929=var12763[15:16]
  #[]
  var12930=tf.reshape(var12929, [])
  #[2500]
  var12931=tf.gather(params=var12762, indices=var12930, batch_dims=0, axis=0)
  #[2500]
  var12932=tf.multiply(var12761, var12931)
  #[50,50]
  var12933=tf.reshape(var12932, [50,50])
  #[1,50]
  var12934=tf.matmul(var12928, var12933)
  #[50]
  var12935=tf.reshape(var12934, [50])
  #[50]
  var12936=tf.multiply(var12743, var12935)
  #[1,50]
  var12937=tf.reshape(var12936, [1,50])
  #[50]
  var12938=tf.multiply(var12747, var12935)
  #[1,50]
  var12939=tf.reshape(var12938, [1,50])
  #[1]
  var12940=var12763[16:17]
  #[]
  var12941=tf.reshape(var12940, [])
  #[2500]
  var12942=tf.gather(params=var12762, indices=var12941, batch_dims=0, axis=0)
  #[2500]
  var12943=tf.multiply(var12761, var12942)
  #[50,50]
  var12944=tf.reshape(var12943, [50,50])
  #[1,50]
  var12945=tf.matmul(var12939, var12944)
  #[50]
  var12946=tf.reshape(var12945, [50])
  #[50]
  var12947=tf.multiply(var12743, var12946)
  #[1,50]
  var12948=tf.reshape(var12947, [1,50])
  #[50]
  var12949=tf.multiply(var12747, var12946)
  #[1,50]
  var12950=tf.reshape(var12949, [1,50])
  #[1]
  var12951=var12763[17:18]
  #[]
  var12952=tf.reshape(var12951, [])
  #[2500]
  var12953=tf.gather(params=var12762, indices=var12952, batch_dims=0, axis=0)
  #[2500]
  var12954=tf.multiply(var12761, var12953)
  #[50,50]
  var12955=tf.reshape(var12954, [50,50])
  #[1,50]
  var12956=tf.matmul(var12950, var12955)
  #[50]
  var12957=tf.reshape(var12956, [50])
  #[50]
  var12958=tf.multiply(var12743, var12957)
  #[1,50]
  var12959=tf.reshape(var12958, [1,50])
  #[50]
  var12960=tf.multiply(var12747, var12957)
  #[1,50]
  var12961=tf.reshape(var12960, [1,50])
  #[1]
  var12962=var12763[18:19]
  #[]
  var12963=tf.reshape(var12962, [])
  #[2500]
  var12964=tf.gather(params=var12762, indices=var12963, batch_dims=0, axis=0)
  #[2500]
  var12965=tf.multiply(var12761, var12964)
  #[50,50]
  var12966=tf.reshape(var12965, [50,50])
  #[1,50]
  var12967=tf.matmul(var12961, var12966)
  #[50]
  var12968=tf.reshape(var12967, [50])
  #[50]
  var12969=tf.multiply(var12743, var12968)
  #[1,50]
  var12970=tf.reshape(var12969, [1,50])
  #[50]
  var12971=tf.multiply(var12747, var12968)
  #[1,50]
  var12972=tf.reshape(var12971, [1,50])
  #[1]
  var12973=var12763[19:20]
  #[]
  var12974=tf.reshape(var12973, [])
  #[2500]
  var12975=tf.gather(params=var12762, indices=var12974, batch_dims=0, axis=0)
  #[2500]
  var12976=tf.multiply(var12761, var12975)
  #[50,50]
  var12977=tf.reshape(var12976, [50,50])
  #[1,50]
  var12978=tf.matmul(var12972, var12977)
  #[50]
  var12979=tf.reshape(var12978, [50])
  #[50]
  var12980=tf.multiply(var12743, var12979)
  #[1,50]
  var12981=tf.reshape(var12980, [1,50])
  #[50]
  var12982=tf.multiply(var12747, var12979)
  #[1,50]
  var12983=tf.reshape(var12982, [1,50])
  #[1]
  var12984=var12763[20:21]
  #[]
  var12985=tf.reshape(var12984, [])
  #[2500]
  var12986=tf.gather(params=var12762, indices=var12985, batch_dims=0, axis=0)
  #[2500]
  var12987=tf.multiply(var12761, var12986)
  #[50,50]
  var12988=tf.reshape(var12987, [50,50])
  #[1,50]
  var12989=tf.matmul(var12983, var12988)
  #[50]
  var12990=tf.reshape(var12989, [50])
  #[50]
  var12991=tf.multiply(var12743, var12990)
  #[1,50]
  var12992=tf.reshape(var12991, [1,50])
  #[21,50]
  var12993=tf.concat([var12772
                     ,var12783
                     ,var12794
                     ,var12805
                     ,var12816
                     ,var12827
                     ,var12838
                     ,var12849
                     ,var12860
                     ,var12871
                     ,var12882
                     ,var12893
                     ,var12904
                     ,var12915
                     ,var12926
                     ,var12937
                     ,var12948
                     ,var12959
                     ,var12970
                     ,var12981
                     ,var12992],
                     axis=0)
  return {"states":var12993}
probeStates = {"function":probeStates_fn
              ,"batched":False
              ,"placeholders":{"x":{"shape":[21],"dtype":tf.int32}}}
@tf.function
def probePreds_fn(training_placeholder, embs, projection_w, projection_bias, x):
  
  #[]
  var12994=training_placeholder
  #[50]
  var12995=tf.random.uniform([50], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[50]
  var12996=tf.floor(var12995)
  #[]
  var12997=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[50]
  var12998=tf.broadcast_to(tf.reshape(var12997, [1]), [50])
  #[50]
  var12999=tf.reshape(var12998, [50])
  #[50]
  var13000=tf.divide(var12996, var12999)
  #[]
  var13001=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[50]
  var13002=tf.broadcast_to(tf.reshape(var13001, [1]), [50])
  #[50]
  var13003=tf.reshape(var13002, [50])
  #[50]
  var13004=tf.cond(var12994, true_fn=lambda: var13000, false_fn=lambda: var13003)
  #[50]
  var13005=tf.random.uniform([50], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[50]
  var13006=tf.floor(var13005)
  #[50]
  var13007=tf.divide(var13006, var12999)
  #[50]
  var13008=tf.cond(var12994, true_fn=lambda: var13007, false_fn=lambda: var13003)
  #[]
  var13009=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var13010=tf.broadcast_to(tf.reshape(var13009, [1]), [1])
  #[]
  var13011=tf.reshape(var13010, [])
  #[50]
  var13012=tf.one_hot(var13011, axis=0, dtype=tf.float32, depth=50)
  #[50]
  var13013=tf.multiply(var13008, var13012)
  #[1,50]
  var13014=tf.reshape(var13013, [1,50])
  #[2500]
  var13015=tf.random.uniform([2500], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[2500]
  var13016=tf.floor(var13015)
  #[2500]
  var13017=tf.broadcast_to(tf.reshape(var12997, [1]), [2500])
  #[2500]
  var13018=tf.reshape(var13017, [2500])
  #[2500]
  var13019=tf.divide(var13016, var13018)
  #[2500]
  var13020=tf.broadcast_to(tf.reshape(var13001, [1]), [2500])
  #[2500]
  var13021=tf.reshape(var13020, [2500])
  #[2500]
  var13022=tf.cond(var12994, true_fn=lambda: var13019, false_fn=lambda: var13021)
  #[12,2500]
  var13023=embs
  #[21]
  var13024=x
  #[1]
  var13025=var13024[0:1]
  #[]
  var13026=tf.reshape(var13025, [])
  #[2500]
  var13027=tf.gather(params=var13023, indices=var13026, batch_dims=0, axis=0)
  #[2500]
  var13028=tf.multiply(var13022, var13027)
  #[50,50]
  var13029=tf.reshape(var13028, [50,50])
  #[1,50]
  var13030=tf.matmul(var13014, var13029)
  #[50]
  var13031=tf.reshape(var13030, [50])
  #[50]
  var13032=tf.multiply(var13004, var13031)
  #[1,50]
  var13033=tf.reshape(var13032, [1,50])
  #[50,12]
  var13034=projection_w
  #[1,12]
  var13035=tf.matmul(var13033, var13034)
  #[12]
  var13036=tf.reshape(var13035, [12])
  #[12]
  var13037=projection_bias
  #[12]
  var13038=tf.add(var13036, var13037)
  #[1,12]
  var13039=tf.reshape(var13038, [1,12])
  #[50]
  var13040=tf.multiply(var13008, var13031)
  #[1,50]
  var13041=tf.reshape(var13040, [1,50])
  #[1]
  var13042=var13024[1:2]
  #[]
  var13043=tf.reshape(var13042, [])
  #[2500]
  var13044=tf.gather(params=var13023, indices=var13043, batch_dims=0, axis=0)
  #[2500]
  var13045=tf.multiply(var13022, var13044)
  #[50,50]
  var13046=tf.reshape(var13045, [50,50])
  #[1,50]
  var13047=tf.matmul(var13041, var13046)
  #[50]
  var13048=tf.reshape(var13047, [50])
  #[50]
  var13049=tf.multiply(var13004, var13048)
  #[1,50]
  var13050=tf.reshape(var13049, [1,50])
  #[1,12]
  var13051=tf.matmul(var13050, var13034)
  #[12]
  var13052=tf.reshape(var13051, [12])
  #[12]
  var13053=tf.add(var13052, var13037)
  #[1,12]
  var13054=tf.reshape(var13053, [1,12])
  #[50]
  var13055=tf.multiply(var13008, var13048)
  #[1,50]
  var13056=tf.reshape(var13055, [1,50])
  #[1]
  var13057=var13024[2:3]
  #[]
  var13058=tf.reshape(var13057, [])
  #[2500]
  var13059=tf.gather(params=var13023, indices=var13058, batch_dims=0, axis=0)
  #[2500]
  var13060=tf.multiply(var13022, var13059)
  #[50,50]
  var13061=tf.reshape(var13060, [50,50])
  #[1,50]
  var13062=tf.matmul(var13056, var13061)
  #[50]
  var13063=tf.reshape(var13062, [50])
  #[50]
  var13064=tf.multiply(var13004, var13063)
  #[1,50]
  var13065=tf.reshape(var13064, [1,50])
  #[1,12]
  var13066=tf.matmul(var13065, var13034)
  #[12]
  var13067=tf.reshape(var13066, [12])
  #[12]
  var13068=tf.add(var13067, var13037)
  #[1,12]
  var13069=tf.reshape(var13068, [1,12])
  #[50]
  var13070=tf.multiply(var13008, var13063)
  #[1,50]
  var13071=tf.reshape(var13070, [1,50])
  #[1]
  var13072=var13024[3:4]
  #[]
  var13073=tf.reshape(var13072, [])
  #[2500]
  var13074=tf.gather(params=var13023, indices=var13073, batch_dims=0, axis=0)
  #[2500]
  var13075=tf.multiply(var13022, var13074)
  #[50,50]
  var13076=tf.reshape(var13075, [50,50])
  #[1,50]
  var13077=tf.matmul(var13071, var13076)
  #[50]
  var13078=tf.reshape(var13077, [50])
  #[50]
  var13079=tf.multiply(var13004, var13078)
  #[1,50]
  var13080=tf.reshape(var13079, [1,50])
  #[1,12]
  var13081=tf.matmul(var13080, var13034)
  #[12]
  var13082=tf.reshape(var13081, [12])
  #[12]
  var13083=tf.add(var13082, var13037)
  #[1,12]
  var13084=tf.reshape(var13083, [1,12])
  #[50]
  var13085=tf.multiply(var13008, var13078)
  #[1,50]
  var13086=tf.reshape(var13085, [1,50])
  #[1]
  var13087=var13024[4:5]
  #[]
  var13088=tf.reshape(var13087, [])
  #[2500]
  var13089=tf.gather(params=var13023, indices=var13088, batch_dims=0, axis=0)
  #[2500]
  var13090=tf.multiply(var13022, var13089)
  #[50,50]
  var13091=tf.reshape(var13090, [50,50])
  #[1,50]
  var13092=tf.matmul(var13086, var13091)
  #[50]
  var13093=tf.reshape(var13092, [50])
  #[50]
  var13094=tf.multiply(var13004, var13093)
  #[1,50]
  var13095=tf.reshape(var13094, [1,50])
  #[1,12]
  var13096=tf.matmul(var13095, var13034)
  #[12]
  var13097=tf.reshape(var13096, [12])
  #[12]
  var13098=tf.add(var13097, var13037)
  #[1,12]
  var13099=tf.reshape(var13098, [1,12])
  #[50]
  var13100=tf.multiply(var13008, var13093)
  #[1,50]
  var13101=tf.reshape(var13100, [1,50])
  #[1]
  var13102=var13024[5:6]
  #[]
  var13103=tf.reshape(var13102, [])
  #[2500]
  var13104=tf.gather(params=var13023, indices=var13103, batch_dims=0, axis=0)
  #[2500]
  var13105=tf.multiply(var13022, var13104)
  #[50,50]
  var13106=tf.reshape(var13105, [50,50])
  #[1,50]
  var13107=tf.matmul(var13101, var13106)
  #[50]
  var13108=tf.reshape(var13107, [50])
  #[50]
  var13109=tf.multiply(var13004, var13108)
  #[1,50]
  var13110=tf.reshape(var13109, [1,50])
  #[1,12]
  var13111=tf.matmul(var13110, var13034)
  #[12]
  var13112=tf.reshape(var13111, [12])
  #[12]
  var13113=tf.add(var13112, var13037)
  #[1,12]
  var13114=tf.reshape(var13113, [1,12])
  #[50]
  var13115=tf.multiply(var13008, var13108)
  #[1,50]
  var13116=tf.reshape(var13115, [1,50])
  #[1]
  var13117=var13024[6:7]
  #[]
  var13118=tf.reshape(var13117, [])
  #[2500]
  var13119=tf.gather(params=var13023, indices=var13118, batch_dims=0, axis=0)
  #[2500]
  var13120=tf.multiply(var13022, var13119)
  #[50,50]
  var13121=tf.reshape(var13120, [50,50])
  #[1,50]
  var13122=tf.matmul(var13116, var13121)
  #[50]
  var13123=tf.reshape(var13122, [50])
  #[50]
  var13124=tf.multiply(var13004, var13123)
  #[1,50]
  var13125=tf.reshape(var13124, [1,50])
  #[1,12]
  var13126=tf.matmul(var13125, var13034)
  #[12]
  var13127=tf.reshape(var13126, [12])
  #[12]
  var13128=tf.add(var13127, var13037)
  #[1,12]
  var13129=tf.reshape(var13128, [1,12])
  #[50]
  var13130=tf.multiply(var13008, var13123)
  #[1,50]
  var13131=tf.reshape(var13130, [1,50])
  #[1]
  var13132=var13024[7:8]
  #[]
  var13133=tf.reshape(var13132, [])
  #[2500]
  var13134=tf.gather(params=var13023, indices=var13133, batch_dims=0, axis=0)
  #[2500]
  var13135=tf.multiply(var13022, var13134)
  #[50,50]
  var13136=tf.reshape(var13135, [50,50])
  #[1,50]
  var13137=tf.matmul(var13131, var13136)
  #[50]
  var13138=tf.reshape(var13137, [50])
  #[50]
  var13139=tf.multiply(var13004, var13138)
  #[1,50]
  var13140=tf.reshape(var13139, [1,50])
  #[1,12]
  var13141=tf.matmul(var13140, var13034)
  #[12]
  var13142=tf.reshape(var13141, [12])
  #[12]
  var13143=tf.add(var13142, var13037)
  #[1,12]
  var13144=tf.reshape(var13143, [1,12])
  #[50]
  var13145=tf.multiply(var13008, var13138)
  #[1,50]
  var13146=tf.reshape(var13145, [1,50])
  #[1]
  var13147=var13024[8:9]
  #[]
  var13148=tf.reshape(var13147, [])
  #[2500]
  var13149=tf.gather(params=var13023, indices=var13148, batch_dims=0, axis=0)
  #[2500]
  var13150=tf.multiply(var13022, var13149)
  #[50,50]
  var13151=tf.reshape(var13150, [50,50])
  #[1,50]
  var13152=tf.matmul(var13146, var13151)
  #[50]
  var13153=tf.reshape(var13152, [50])
  #[50]
  var13154=tf.multiply(var13004, var13153)
  #[1,50]
  var13155=tf.reshape(var13154, [1,50])
  #[1,12]
  var13156=tf.matmul(var13155, var13034)
  #[12]
  var13157=tf.reshape(var13156, [12])
  #[12]
  var13158=tf.add(var13157, var13037)
  #[1,12]
  var13159=tf.reshape(var13158, [1,12])
  #[50]
  var13160=tf.multiply(var13008, var13153)
  #[1,50]
  var13161=tf.reshape(var13160, [1,50])
  #[1]
  var13162=var13024[9:10]
  #[]
  var13163=tf.reshape(var13162, [])
  #[2500]
  var13164=tf.gather(params=var13023, indices=var13163, batch_dims=0, axis=0)
  #[2500]
  var13165=tf.multiply(var13022, var13164)
  #[50,50]
  var13166=tf.reshape(var13165, [50,50])
  #[1,50]
  var13167=tf.matmul(var13161, var13166)
  #[50]
  var13168=tf.reshape(var13167, [50])
  #[50]
  var13169=tf.multiply(var13004, var13168)
  #[1,50]
  var13170=tf.reshape(var13169, [1,50])
  #[1,12]
  var13171=tf.matmul(var13170, var13034)
  #[12]
  var13172=tf.reshape(var13171, [12])
  #[12]
  var13173=tf.add(var13172, var13037)
  #[1,12]
  var13174=tf.reshape(var13173, [1,12])
  #[50]
  var13175=tf.multiply(var13008, var13168)
  #[1,50]
  var13176=tf.reshape(var13175, [1,50])
  #[1]
  var13177=var13024[10:11]
  #[]
  var13178=tf.reshape(var13177, [])
  #[2500]
  var13179=tf.gather(params=var13023, indices=var13178, batch_dims=0, axis=0)
  #[2500]
  var13180=tf.multiply(var13022, var13179)
  #[50,50]
  var13181=tf.reshape(var13180, [50,50])
  #[1,50]
  var13182=tf.matmul(var13176, var13181)
  #[50]
  var13183=tf.reshape(var13182, [50])
  #[50]
  var13184=tf.multiply(var13004, var13183)
  #[1,50]
  var13185=tf.reshape(var13184, [1,50])
  #[1,12]
  var13186=tf.matmul(var13185, var13034)
  #[12]
  var13187=tf.reshape(var13186, [12])
  #[12]
  var13188=tf.add(var13187, var13037)
  #[1,12]
  var13189=tf.reshape(var13188, [1,12])
  #[50]
  var13190=tf.multiply(var13008, var13183)
  #[1,50]
  var13191=tf.reshape(var13190, [1,50])
  #[1]
  var13192=var13024[11:12]
  #[]
  var13193=tf.reshape(var13192, [])
  #[2500]
  var13194=tf.gather(params=var13023, indices=var13193, batch_dims=0, axis=0)
  #[2500]
  var13195=tf.multiply(var13022, var13194)
  #[50,50]
  var13196=tf.reshape(var13195, [50,50])
  #[1,50]
  var13197=tf.matmul(var13191, var13196)
  #[50]
  var13198=tf.reshape(var13197, [50])
  #[50]
  var13199=tf.multiply(var13004, var13198)
  #[1,50]
  var13200=tf.reshape(var13199, [1,50])
  #[1,12]
  var13201=tf.matmul(var13200, var13034)
  #[12]
  var13202=tf.reshape(var13201, [12])
  #[12]
  var13203=tf.add(var13202, var13037)
  #[1,12]
  var13204=tf.reshape(var13203, [1,12])
  #[50]
  var13205=tf.multiply(var13008, var13198)
  #[1,50]
  var13206=tf.reshape(var13205, [1,50])
  #[1]
  var13207=var13024[12:13]
  #[]
  var13208=tf.reshape(var13207, [])
  #[2500]
  var13209=tf.gather(params=var13023, indices=var13208, batch_dims=0, axis=0)
  #[2500]
  var13210=tf.multiply(var13022, var13209)
  #[50,50]
  var13211=tf.reshape(var13210, [50,50])
  #[1,50]
  var13212=tf.matmul(var13206, var13211)
  #[50]
  var13213=tf.reshape(var13212, [50])
  #[50]
  var13214=tf.multiply(var13004, var13213)
  #[1,50]
  var13215=tf.reshape(var13214, [1,50])
  #[1,12]
  var13216=tf.matmul(var13215, var13034)
  #[12]
  var13217=tf.reshape(var13216, [12])
  #[12]
  var13218=tf.add(var13217, var13037)
  #[1,12]
  var13219=tf.reshape(var13218, [1,12])
  #[50]
  var13220=tf.multiply(var13008, var13213)
  #[1,50]
  var13221=tf.reshape(var13220, [1,50])
  #[1]
  var13222=var13024[13:14]
  #[]
  var13223=tf.reshape(var13222, [])
  #[2500]
  var13224=tf.gather(params=var13023, indices=var13223, batch_dims=0, axis=0)
  #[2500]
  var13225=tf.multiply(var13022, var13224)
  #[50,50]
  var13226=tf.reshape(var13225, [50,50])
  #[1,50]
  var13227=tf.matmul(var13221, var13226)
  #[50]
  var13228=tf.reshape(var13227, [50])
  #[50]
  var13229=tf.multiply(var13004, var13228)
  #[1,50]
  var13230=tf.reshape(var13229, [1,50])
  #[1,12]
  var13231=tf.matmul(var13230, var13034)
  #[12]
  var13232=tf.reshape(var13231, [12])
  #[12]
  var13233=tf.add(var13232, var13037)
  #[1,12]
  var13234=tf.reshape(var13233, [1,12])
  #[50]
  var13235=tf.multiply(var13008, var13228)
  #[1,50]
  var13236=tf.reshape(var13235, [1,50])
  #[1]
  var13237=var13024[14:15]
  #[]
  var13238=tf.reshape(var13237, [])
  #[2500]
  var13239=tf.gather(params=var13023, indices=var13238, batch_dims=0, axis=0)
  #[2500]
  var13240=tf.multiply(var13022, var13239)
  #[50,50]
  var13241=tf.reshape(var13240, [50,50])
  #[1,50]
  var13242=tf.matmul(var13236, var13241)
  #[50]
  var13243=tf.reshape(var13242, [50])
  #[50]
  var13244=tf.multiply(var13004, var13243)
  #[1,50]
  var13245=tf.reshape(var13244, [1,50])
  #[1,12]
  var13246=tf.matmul(var13245, var13034)
  #[12]
  var13247=tf.reshape(var13246, [12])
  #[12]
  var13248=tf.add(var13247, var13037)
  #[1,12]
  var13249=tf.reshape(var13248, [1,12])
  #[50]
  var13250=tf.multiply(var13008, var13243)
  #[1,50]
  var13251=tf.reshape(var13250, [1,50])
  #[1]
  var13252=var13024[15:16]
  #[]
  var13253=tf.reshape(var13252, [])
  #[2500]
  var13254=tf.gather(params=var13023, indices=var13253, batch_dims=0, axis=0)
  #[2500]
  var13255=tf.multiply(var13022, var13254)
  #[50,50]
  var13256=tf.reshape(var13255, [50,50])
  #[1,50]
  var13257=tf.matmul(var13251, var13256)
  #[50]
  var13258=tf.reshape(var13257, [50])
  #[50]
  var13259=tf.multiply(var13004, var13258)
  #[1,50]
  var13260=tf.reshape(var13259, [1,50])
  #[1,12]
  var13261=tf.matmul(var13260, var13034)
  #[12]
  var13262=tf.reshape(var13261, [12])
  #[12]
  var13263=tf.add(var13262, var13037)
  #[1,12]
  var13264=tf.reshape(var13263, [1,12])
  #[50]
  var13265=tf.multiply(var13008, var13258)
  #[1,50]
  var13266=tf.reshape(var13265, [1,50])
  #[1]
  var13267=var13024[16:17]
  #[]
  var13268=tf.reshape(var13267, [])
  #[2500]
  var13269=tf.gather(params=var13023, indices=var13268, batch_dims=0, axis=0)
  #[2500]
  var13270=tf.multiply(var13022, var13269)
  #[50,50]
  var13271=tf.reshape(var13270, [50,50])
  #[1,50]
  var13272=tf.matmul(var13266, var13271)
  #[50]
  var13273=tf.reshape(var13272, [50])
  #[50]
  var13274=tf.multiply(var13004, var13273)
  #[1,50]
  var13275=tf.reshape(var13274, [1,50])
  #[1,12]
  var13276=tf.matmul(var13275, var13034)
  #[12]
  var13277=tf.reshape(var13276, [12])
  #[12]
  var13278=tf.add(var13277, var13037)
  #[1,12]
  var13279=tf.reshape(var13278, [1,12])
  #[50]
  var13280=tf.multiply(var13008, var13273)
  #[1,50]
  var13281=tf.reshape(var13280, [1,50])
  #[1]
  var13282=var13024[17:18]
  #[]
  var13283=tf.reshape(var13282, [])
  #[2500]
  var13284=tf.gather(params=var13023, indices=var13283, batch_dims=0, axis=0)
  #[2500]
  var13285=tf.multiply(var13022, var13284)
  #[50,50]
  var13286=tf.reshape(var13285, [50,50])
  #[1,50]
  var13287=tf.matmul(var13281, var13286)
  #[50]
  var13288=tf.reshape(var13287, [50])
  #[50]
  var13289=tf.multiply(var13004, var13288)
  #[1,50]
  var13290=tf.reshape(var13289, [1,50])
  #[1,12]
  var13291=tf.matmul(var13290, var13034)
  #[12]
  var13292=tf.reshape(var13291, [12])
  #[12]
  var13293=tf.add(var13292, var13037)
  #[1,12]
  var13294=tf.reshape(var13293, [1,12])
  #[50]
  var13295=tf.multiply(var13008, var13288)
  #[1,50]
  var13296=tf.reshape(var13295, [1,50])
  #[1]
  var13297=var13024[18:19]
  #[]
  var13298=tf.reshape(var13297, [])
  #[2500]
  var13299=tf.gather(params=var13023, indices=var13298, batch_dims=0, axis=0)
  #[2500]
  var13300=tf.multiply(var13022, var13299)
  #[50,50]
  var13301=tf.reshape(var13300, [50,50])
  #[1,50]
  var13302=tf.matmul(var13296, var13301)
  #[50]
  var13303=tf.reshape(var13302, [50])
  #[50]
  var13304=tf.multiply(var13004, var13303)
  #[1,50]
  var13305=tf.reshape(var13304, [1,50])
  #[1,12]
  var13306=tf.matmul(var13305, var13034)
  #[12]
  var13307=tf.reshape(var13306, [12])
  #[12]
  var13308=tf.add(var13307, var13037)
  #[1,12]
  var13309=tf.reshape(var13308, [1,12])
  #[50]
  var13310=tf.multiply(var13008, var13303)
  #[1,50]
  var13311=tf.reshape(var13310, [1,50])
  #[1]
  var13312=var13024[19:20]
  #[]
  var13313=tf.reshape(var13312, [])
  #[2500]
  var13314=tf.gather(params=var13023, indices=var13313, batch_dims=0, axis=0)
  #[2500]
  var13315=tf.multiply(var13022, var13314)
  #[50,50]
  var13316=tf.reshape(var13315, [50,50])
  #[1,50]
  var13317=tf.matmul(var13311, var13316)
  #[50]
  var13318=tf.reshape(var13317, [50])
  #[50]
  var13319=tf.multiply(var13004, var13318)
  #[1,50]
  var13320=tf.reshape(var13319, [1,50])
  #[1,12]
  var13321=tf.matmul(var13320, var13034)
  #[12]
  var13322=tf.reshape(var13321, [12])
  #[12]
  var13323=tf.add(var13322, var13037)
  #[1,12]
  var13324=tf.reshape(var13323, [1,12])
  #[50]
  var13325=tf.multiply(var13008, var13318)
  #[1,50]
  var13326=tf.reshape(var13325, [1,50])
  #[1]
  var13327=var13024[20:21]
  #[]
  var13328=tf.reshape(var13327, [])
  #[2500]
  var13329=tf.gather(params=var13023, indices=var13328, batch_dims=0, axis=0)
  #[2500]
  var13330=tf.multiply(var13022, var13329)
  #[50,50]
  var13331=tf.reshape(var13330, [50,50])
  #[1,50]
  var13332=tf.matmul(var13326, var13331)
  #[50]
  var13333=tf.reshape(var13332, [50])
  #[50]
  var13334=tf.multiply(var13004, var13333)
  #[1,50]
  var13335=tf.reshape(var13334, [1,50])
  #[1,12]
  var13336=tf.matmul(var13335, var13034)
  #[12]
  var13337=tf.reshape(var13336, [12])
  #[12]
  var13338=tf.add(var13337, var13037)
  #[1,12]
  var13339=tf.reshape(var13338, [1,12])
  #[21,12]
  var13340=tf.concat([var13039
                     ,var13054
                     ,var13069
                     ,var13084
                     ,var13099
                     ,var13114
                     ,var13129
                     ,var13144
                     ,var13159
                     ,var13174
                     ,var13189
                     ,var13204
                     ,var13219
                     ,var13234
                     ,var13249
                     ,var13264
                     ,var13279
                     ,var13294
                     ,var13309
                     ,var13324
                     ,var13339],
                     axis=0)
  #[21]
  var13341=tf.argmax(var13340, axis=1, output_type=tf.int32)
  return {"pred":var13340,"y":var13341}
probePreds = {"function":probePreds_fn
             ,"batched":False
             ,"placeholders":{"x":{"shape":[21],"dtype":tf.int32}}}