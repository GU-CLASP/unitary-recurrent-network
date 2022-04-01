
import tensorflow as tf
def mkModel():
  
  #[12,160]
  var12345=tf.random.uniform(
             [12,160], minval=-1.0471976, maxval=1.0471976, dtype=tf.float32) # 0
  var12346=tf.Variable(name="embs", trainable=True, initial_value=var12345)
  #[65,12]
  var12347=tf.random.uniform(
             [65,12], minval=-0.27914527, maxval=0.27914527, dtype=tf.float32) # 4
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
  #[512,65]
  var12353=tf.random.uniform([512,65], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[512,65]
  var12354=tf.floor(var12353)
  #[]
  var12355=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[65]
  var12356=tf.broadcast_to(tf.reshape(var12355, [1]), [65])
  #[65]
  var12357=tf.reshape(var12356, [65])
  #[512,65]
  var12358=tf.broadcast_to(tf.reshape(var12357, [1,65]), [512,65])
  #[512,65]
  var12359=tf.divide(var12354, var12358)
  #[]
  var12360=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[65]
  var12361=tf.broadcast_to(tf.reshape(var12360, [1]), [65])
  #[65]
  var12362=tf.reshape(var12361, [65])
  #[512,65]
  var12363=tf.broadcast_to(tf.reshape(var12362, [1,65]), [512,65])
  #[512,65]
  var12364=tf.cond(var12352, true_fn=lambda: var12359, false_fn=lambda: var12363)
  #[512,65]
  var12365=tf.random.uniform([512,65], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[512,65]
  var12366=tf.floor(var12365)
  #[512,65]
  var12367=tf.divide(var12366, var12358)
  #[512,65]
  var12368=tf.cond(var12352, true_fn=lambda: var12367, false_fn=lambda: var12363)
  #[]
  var12369=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var12370=tf.broadcast_to(tf.reshape(var12369, [1]), [1])
  #[]
  var12371=tf.reshape(var12370, [])
  #[65]
  var12372=tf.one_hot(var12371, axis=0, dtype=tf.float32, depth=65)
  #[512,65]
  var12373=tf.broadcast_to(tf.reshape(var12372, [1,65]), [512,65])
  #[512,65]
  var12374=tf.multiply(var12368, var12373)
  #[512,1]
  var12375=var12374[:,0:1]
  #[512,1]
  var12376=tf.reshape(var12375, [512,1])
  #[512,64]
  var12377=var12374[:,1:65]
  #[512,32,1,2]
  var12378=tf.reshape(var12377, [512,32,1,2])
  #[512,160]
  var12379=tf.random.uniform([512,160], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[512,160]
  var12380=tf.floor(var12379)
  #[160]
  var12381=tf.broadcast_to(tf.reshape(var12355, [1]), [160])
  #[160]
  var12382=tf.reshape(var12381, [160])
  #[512,160]
  var12383=tf.broadcast_to(tf.reshape(var12382, [1,160]), [512,160])
  #[512,160]
  var12384=tf.divide(var12380, var12383)
  #[160]
  var12385=tf.broadcast_to(tf.reshape(var12360, [1]), [160])
  #[160]
  var12386=tf.reshape(var12385, [160])
  #[512,160]
  var12387=tf.broadcast_to(tf.reshape(var12386, [1,160]), [512,160])
  #[512,160]
  var12388=tf.cond(var12352, true_fn=lambda: var12384, false_fn=lambda: var12387)
  #[12,160]
  var12389=embs
  #[512,21]
  var12390=x
  #[512,1]
  var12391=var12390[:,0:1]
  #[512]
  var12392=tf.reshape(var12391, [512])
  #[512,160]
  var12393=tf.gather(params=var12389, indices=var12392, batch_dims=0, axis=0)
  #[512,160]
  var12394=tf.multiply(var12388, var12393)
  #[512,5,32]
  var12395=tf.reshape(var12394, [512,5,32])
  #[512,1,32]
  var12396=var12395[:,4:5]
  #[512,32]
  var12397=tf.reshape(var12396, [512,32])
  #[512,32]
  var12398=tf.cos(var12397)
  #[512,32,1]
  var12399=tf.reshape(var12398, [512,32,1])
  #[512,32]
  var12400=tf.sin(var12397)
  #[512,32]
  var12401=tf.negative(var12400)
  #[512,32,1]
  var12402=tf.reshape(var12401, [512,32,1])
  #[512,32,2]
  var12403=tf.concat([var12399,var12402], axis=2)
  #[512,32,1,2]
  var12404=tf.reshape(var12403, [512,32,1,2])
  #[512,32,1]
  var12405=tf.reshape(var12400, [512,32,1])
  #[512,32,2]
  var12406=tf.concat([var12405,var12399], axis=2)
  #[512,32,1,2]
  var12407=tf.reshape(var12406, [512,32,1,2])
  #[512,32,2,2]
  var12408=tf.concat([var12404,var12407], axis=2)
  #[512,32,1,2]
  var12409=tf.matmul(var12378, var12408)
  #[512,64]
  var12410=tf.reshape(var12409, [512,64])
  #[512,65]
  var12411=tf.concat([var12376,var12410], axis=1)
  #[512,64]
  var12412=var12411[:,0:64]
  #[512,32,1,2]
  var12413=tf.reshape(var12412, [512,32,1,2])
  #[512,1,32]
  var12414=var12395[:,3:4]
  #[512,32]
  var12415=tf.reshape(var12414, [512,32])
  #[512,32]
  var12416=tf.cos(var12415)
  #[512,32,1]
  var12417=tf.reshape(var12416, [512,32,1])
  #[512,32]
  var12418=tf.sin(var12415)
  #[512,32]
  var12419=tf.negative(var12418)
  #[512,32,1]
  var12420=tf.reshape(var12419, [512,32,1])
  #[512,32,2]
  var12421=tf.concat([var12417,var12420], axis=2)
  #[512,32,1,2]
  var12422=tf.reshape(var12421, [512,32,1,2])
  #[512,32,1]
  var12423=tf.reshape(var12418, [512,32,1])
  #[512,32,2]
  var12424=tf.concat([var12423,var12417], axis=2)
  #[512,32,1,2]
  var12425=tf.reshape(var12424, [512,32,1,2])
  #[512,32,2,2]
  var12426=tf.concat([var12422,var12425], axis=2)
  #[512,32,1,2]
  var12427=tf.matmul(var12413, var12426)
  #[512,64]
  var12428=tf.reshape(var12427, [512,64])
  #[512,1]
  var12429=var12411[:,64:65]
  #[512,1]
  var12430=tf.reshape(var12429, [512,1])
  #[512,65]
  var12431=tf.concat([var12428,var12430], axis=1)
  #[512,1]
  var12432=var12431[:,0:1]
  #[512,1]
  var12433=tf.reshape(var12432, [512,1])
  #[512,64]
  var12434=var12431[:,1:65]
  #[512,32,1,2]
  var12435=tf.reshape(var12434, [512,32,1,2])
  #[512,1,32]
  var12436=var12395[:,2:3]
  #[512,32]
  var12437=tf.reshape(var12436, [512,32])
  #[512,32]
  var12438=tf.cos(var12437)
  #[512,32,1]
  var12439=tf.reshape(var12438, [512,32,1])
  #[512,32]
  var12440=tf.sin(var12437)
  #[512,32]
  var12441=tf.negative(var12440)
  #[512,32,1]
  var12442=tf.reshape(var12441, [512,32,1])
  #[512,32,2]
  var12443=tf.concat([var12439,var12442], axis=2)
  #[512,32,1,2]
  var12444=tf.reshape(var12443, [512,32,1,2])
  #[512,32,1]
  var12445=tf.reshape(var12440, [512,32,1])
  #[512,32,2]
  var12446=tf.concat([var12445,var12439], axis=2)
  #[512,32,1,2]
  var12447=tf.reshape(var12446, [512,32,1,2])
  #[512,32,2,2]
  var12448=tf.concat([var12444,var12447], axis=2)
  #[512,32,1,2]
  var12449=tf.matmul(var12435, var12448)
  #[512,64]
  var12450=tf.reshape(var12449, [512,64])
  #[512,65]
  var12451=tf.concat([var12433,var12450], axis=1)
  #[512,64]
  var12452=var12451[:,0:64]
  #[512,32,1,2]
  var12453=tf.reshape(var12452, [512,32,1,2])
  #[512,1,32]
  var12454=var12395[:,1:2]
  #[512,32]
  var12455=tf.reshape(var12454, [512,32])
  #[512,32]
  var12456=tf.cos(var12455)
  #[512,32,1]
  var12457=tf.reshape(var12456, [512,32,1])
  #[512,32]
  var12458=tf.sin(var12455)
  #[512,32]
  var12459=tf.negative(var12458)
  #[512,32,1]
  var12460=tf.reshape(var12459, [512,32,1])
  #[512,32,2]
  var12461=tf.concat([var12457,var12460], axis=2)
  #[512,32,1,2]
  var12462=tf.reshape(var12461, [512,32,1,2])
  #[512,32,1]
  var12463=tf.reshape(var12458, [512,32,1])
  #[512,32,2]
  var12464=tf.concat([var12463,var12457], axis=2)
  #[512,32,1,2]
  var12465=tf.reshape(var12464, [512,32,1,2])
  #[512,32,2,2]
  var12466=tf.concat([var12462,var12465], axis=2)
  #[512,32,1,2]
  var12467=tf.matmul(var12453, var12466)
  #[512,64]
  var12468=tf.reshape(var12467, [512,64])
  #[512,1]
  var12469=var12451[:,64:65]
  #[512,1]
  var12470=tf.reshape(var12469, [512,1])
  #[512,65]
  var12471=tf.concat([var12468,var12470], axis=1)
  #[512,1]
  var12472=var12471[:,0:1]
  #[512,1]
  var12473=tf.reshape(var12472, [512,1])
  #[512,64]
  var12474=var12471[:,1:65]
  #[512,32,1,2]
  var12475=tf.reshape(var12474, [512,32,1,2])
  #[512,1,32]
  var12476=var12395[:,0:1]
  #[512,32]
  var12477=tf.reshape(var12476, [512,32])
  #[512,32]
  var12478=tf.cos(var12477)
  #[512,32,1]
  var12479=tf.reshape(var12478, [512,32,1])
  #[512,32]
  var12480=tf.sin(var12477)
  #[512,32]
  var12481=tf.negative(var12480)
  #[512,32,1]
  var12482=tf.reshape(var12481, [512,32,1])
  #[512,32,2]
  var12483=tf.concat([var12479,var12482], axis=2)
  #[512,32,1,2]
  var12484=tf.reshape(var12483, [512,32,1,2])
  #[512,32,1]
  var12485=tf.reshape(var12480, [512,32,1])
  #[512,32,2]
  var12486=tf.concat([var12485,var12479], axis=2)
  #[512,32,1,2]
  var12487=tf.reshape(var12486, [512,32,1,2])
  #[512,32,2,2]
  var12488=tf.concat([var12484,var12487], axis=2)
  #[512,32,1,2]
  var12489=tf.matmul(var12475, var12488)
  #[512,64]
  var12490=tf.reshape(var12489, [512,64])
  #[512,65]
  var12491=tf.concat([var12473,var12490], axis=1)
  #[512,65]
  var12492=tf.multiply(var12364, var12491)
  #[512,65]
  var12493=tf.reshape(var12492, [512,65])
  #[65,12]
  var12494=projection_w
  #[512,12]
  var12495=tf.matmul(var12493, var12494)
  #[512,12]
  var12496=tf.reshape(var12495, [512,12])
  #[12]
  var12497=projection_bias
  #[512,12]
  var12498=tf.broadcast_to(tf.reshape(var12497, [1,12]), [512,12])
  #[512,12]
  var12499=tf.add(var12496, var12498)
  #[512,1,12]
  var12500=tf.reshape(var12499, [512,1,12])
  #[512,65]
  var12501=tf.multiply(var12368, var12491)
  #[512,1]
  var12502=var12501[:,0:1]
  #[512,1]
  var12503=tf.reshape(var12502, [512,1])
  #[512,64]
  var12504=var12501[:,1:65]
  #[512,32,1,2]
  var12505=tf.reshape(var12504, [512,32,1,2])
  #[512,1]
  var12506=var12390[:,1:2]
  #[512]
  var12507=tf.reshape(var12506, [512])
  #[512,160]
  var12508=tf.gather(params=var12389, indices=var12507, batch_dims=0, axis=0)
  #[512,160]
  var12509=tf.multiply(var12388, var12508)
  #[512,5,32]
  var12510=tf.reshape(var12509, [512,5,32])
  #[512,1,32]
  var12511=var12510[:,4:5]
  #[512,32]
  var12512=tf.reshape(var12511, [512,32])
  #[512,32]
  var12513=tf.cos(var12512)
  #[512,32,1]
  var12514=tf.reshape(var12513, [512,32,1])
  #[512,32]
  var12515=tf.sin(var12512)
  #[512,32]
  var12516=tf.negative(var12515)
  #[512,32,1]
  var12517=tf.reshape(var12516, [512,32,1])
  #[512,32,2]
  var12518=tf.concat([var12514,var12517], axis=2)
  #[512,32,1,2]
  var12519=tf.reshape(var12518, [512,32,1,2])
  #[512,32,1]
  var12520=tf.reshape(var12515, [512,32,1])
  #[512,32,2]
  var12521=tf.concat([var12520,var12514], axis=2)
  #[512,32,1,2]
  var12522=tf.reshape(var12521, [512,32,1,2])
  #[512,32,2,2]
  var12523=tf.concat([var12519,var12522], axis=2)
  #[512,32,1,2]
  var12524=tf.matmul(var12505, var12523)
  #[512,64]
  var12525=tf.reshape(var12524, [512,64])
  #[512,65]
  var12526=tf.concat([var12503,var12525], axis=1)
  #[512,64]
  var12527=var12526[:,0:64]
  #[512,32,1,2]
  var12528=tf.reshape(var12527, [512,32,1,2])
  #[512,1,32]
  var12529=var12510[:,3:4]
  #[512,32]
  var12530=tf.reshape(var12529, [512,32])
  #[512,32]
  var12531=tf.cos(var12530)
  #[512,32,1]
  var12532=tf.reshape(var12531, [512,32,1])
  #[512,32]
  var12533=tf.sin(var12530)
  #[512,32]
  var12534=tf.negative(var12533)
  #[512,32,1]
  var12535=tf.reshape(var12534, [512,32,1])
  #[512,32,2]
  var12536=tf.concat([var12532,var12535], axis=2)
  #[512,32,1,2]
  var12537=tf.reshape(var12536, [512,32,1,2])
  #[512,32,1]
  var12538=tf.reshape(var12533, [512,32,1])
  #[512,32,2]
  var12539=tf.concat([var12538,var12532], axis=2)
  #[512,32,1,2]
  var12540=tf.reshape(var12539, [512,32,1,2])
  #[512,32,2,2]
  var12541=tf.concat([var12537,var12540], axis=2)
  #[512,32,1,2]
  var12542=tf.matmul(var12528, var12541)
  #[512,64]
  var12543=tf.reshape(var12542, [512,64])
  #[512,1]
  var12544=var12526[:,64:65]
  #[512,1]
  var12545=tf.reshape(var12544, [512,1])
  #[512,65]
  var12546=tf.concat([var12543,var12545], axis=1)
  #[512,1]
  var12547=var12546[:,0:1]
  #[512,1]
  var12548=tf.reshape(var12547, [512,1])
  #[512,64]
  var12549=var12546[:,1:65]
  #[512,32,1,2]
  var12550=tf.reshape(var12549, [512,32,1,2])
  #[512,1,32]
  var12551=var12510[:,2:3]
  #[512,32]
  var12552=tf.reshape(var12551, [512,32])
  #[512,32]
  var12553=tf.cos(var12552)
  #[512,32,1]
  var12554=tf.reshape(var12553, [512,32,1])
  #[512,32]
  var12555=tf.sin(var12552)
  #[512,32]
  var12556=tf.negative(var12555)
  #[512,32,1]
  var12557=tf.reshape(var12556, [512,32,1])
  #[512,32,2]
  var12558=tf.concat([var12554,var12557], axis=2)
  #[512,32,1,2]
  var12559=tf.reshape(var12558, [512,32,1,2])
  #[512,32,1]
  var12560=tf.reshape(var12555, [512,32,1])
  #[512,32,2]
  var12561=tf.concat([var12560,var12554], axis=2)
  #[512,32,1,2]
  var12562=tf.reshape(var12561, [512,32,1,2])
  #[512,32,2,2]
  var12563=tf.concat([var12559,var12562], axis=2)
  #[512,32,1,2]
  var12564=tf.matmul(var12550, var12563)
  #[512,64]
  var12565=tf.reshape(var12564, [512,64])
  #[512,65]
  var12566=tf.concat([var12548,var12565], axis=1)
  #[512,64]
  var12567=var12566[:,0:64]
  #[512,32,1,2]
  var12568=tf.reshape(var12567, [512,32,1,2])
  #[512,1,32]
  var12569=var12510[:,1:2]
  #[512,32]
  var12570=tf.reshape(var12569, [512,32])
  #[512,32]
  var12571=tf.cos(var12570)
  #[512,32,1]
  var12572=tf.reshape(var12571, [512,32,1])
  #[512,32]
  var12573=tf.sin(var12570)
  #[512,32]
  var12574=tf.negative(var12573)
  #[512,32,1]
  var12575=tf.reshape(var12574, [512,32,1])
  #[512,32,2]
  var12576=tf.concat([var12572,var12575], axis=2)
  #[512,32,1,2]
  var12577=tf.reshape(var12576, [512,32,1,2])
  #[512,32,1]
  var12578=tf.reshape(var12573, [512,32,1])
  #[512,32,2]
  var12579=tf.concat([var12578,var12572], axis=2)
  #[512,32,1,2]
  var12580=tf.reshape(var12579, [512,32,1,2])
  #[512,32,2,2]
  var12581=tf.concat([var12577,var12580], axis=2)
  #[512,32,1,2]
  var12582=tf.matmul(var12568, var12581)
  #[512,64]
  var12583=tf.reshape(var12582, [512,64])
  #[512,1]
  var12584=var12566[:,64:65]
  #[512,1]
  var12585=tf.reshape(var12584, [512,1])
  #[512,65]
  var12586=tf.concat([var12583,var12585], axis=1)
  #[512,1]
  var12587=var12586[:,0:1]
  #[512,1]
  var12588=tf.reshape(var12587, [512,1])
  #[512,64]
  var12589=var12586[:,1:65]
  #[512,32,1,2]
  var12590=tf.reshape(var12589, [512,32,1,2])
  #[512,1,32]
  var12591=var12510[:,0:1]
  #[512,32]
  var12592=tf.reshape(var12591, [512,32])
  #[512,32]
  var12593=tf.cos(var12592)
  #[512,32,1]
  var12594=tf.reshape(var12593, [512,32,1])
  #[512,32]
  var12595=tf.sin(var12592)
  #[512,32]
  var12596=tf.negative(var12595)
  #[512,32,1]
  var12597=tf.reshape(var12596, [512,32,1])
  #[512,32,2]
  var12598=tf.concat([var12594,var12597], axis=2)
  #[512,32,1,2]
  var12599=tf.reshape(var12598, [512,32,1,2])
  #[512,32,1]
  var12600=tf.reshape(var12595, [512,32,1])
  #[512,32,2]
  var12601=tf.concat([var12600,var12594], axis=2)
  #[512,32,1,2]
  var12602=tf.reshape(var12601, [512,32,1,2])
  #[512,32,2,2]
  var12603=tf.concat([var12599,var12602], axis=2)
  #[512,32,1,2]
  var12604=tf.matmul(var12590, var12603)
  #[512,64]
  var12605=tf.reshape(var12604, [512,64])
  #[512,65]
  var12606=tf.concat([var12588,var12605], axis=1)
  #[512,65]
  var12607=tf.multiply(var12364, var12606)
  #[512,65]
  var12608=tf.reshape(var12607, [512,65])
  #[512,12]
  var12609=tf.matmul(var12608, var12494)
  #[512,12]
  var12610=tf.reshape(var12609, [512,12])
  #[512,12]
  var12611=tf.add(var12610, var12498)
  #[512,1,12]
  var12612=tf.reshape(var12611, [512,1,12])
  #[512,65]
  var12613=tf.multiply(var12368, var12606)
  #[512,1]
  var12614=var12613[:,0:1]
  #[512,1]
  var12615=tf.reshape(var12614, [512,1])
  #[512,64]
  var12616=var12613[:,1:65]
  #[512,32,1,2]
  var12617=tf.reshape(var12616, [512,32,1,2])
  #[512,1]
  var12618=var12390[:,2:3]
  #[512]
  var12619=tf.reshape(var12618, [512])
  #[512,160]
  var12620=tf.gather(params=var12389, indices=var12619, batch_dims=0, axis=0)
  #[512,160]
  var12621=tf.multiply(var12388, var12620)
  #[512,5,32]
  var12622=tf.reshape(var12621, [512,5,32])
  #[512,1,32]
  var12623=var12622[:,4:5]
  #[512,32]
  var12624=tf.reshape(var12623, [512,32])
  #[512,32]
  var12625=tf.cos(var12624)
  #[512,32,1]
  var12626=tf.reshape(var12625, [512,32,1])
  #[512,32]
  var12627=tf.sin(var12624)
  #[512,32]
  var12628=tf.negative(var12627)
  #[512,32,1]
  var12629=tf.reshape(var12628, [512,32,1])
  #[512,32,2]
  var12630=tf.concat([var12626,var12629], axis=2)
  #[512,32,1,2]
  var12631=tf.reshape(var12630, [512,32,1,2])
  #[512,32,1]
  var12632=tf.reshape(var12627, [512,32,1])
  #[512,32,2]
  var12633=tf.concat([var12632,var12626], axis=2)
  #[512,32,1,2]
  var12634=tf.reshape(var12633, [512,32,1,2])
  #[512,32,2,2]
  var12635=tf.concat([var12631,var12634], axis=2)
  #[512,32,1,2]
  var12636=tf.matmul(var12617, var12635)
  #[512,64]
  var12637=tf.reshape(var12636, [512,64])
  #[512,65]
  var12638=tf.concat([var12615,var12637], axis=1)
  #[512,64]
  var12639=var12638[:,0:64]
  #[512,32,1,2]
  var12640=tf.reshape(var12639, [512,32,1,2])
  #[512,1,32]
  var12641=var12622[:,3:4]
  #[512,32]
  var12642=tf.reshape(var12641, [512,32])
  #[512,32]
  var12643=tf.cos(var12642)
  #[512,32,1]
  var12644=tf.reshape(var12643, [512,32,1])
  #[512,32]
  var12645=tf.sin(var12642)
  #[512,32]
  var12646=tf.negative(var12645)
  #[512,32,1]
  var12647=tf.reshape(var12646, [512,32,1])
  #[512,32,2]
  var12648=tf.concat([var12644,var12647], axis=2)
  #[512,32,1,2]
  var12649=tf.reshape(var12648, [512,32,1,2])
  #[512,32,1]
  var12650=tf.reshape(var12645, [512,32,1])
  #[512,32,2]
  var12651=tf.concat([var12650,var12644], axis=2)
  #[512,32,1,2]
  var12652=tf.reshape(var12651, [512,32,1,2])
  #[512,32,2,2]
  var12653=tf.concat([var12649,var12652], axis=2)
  #[512,32,1,2]
  var12654=tf.matmul(var12640, var12653)
  #[512,64]
  var12655=tf.reshape(var12654, [512,64])
  #[512,1]
  var12656=var12638[:,64:65]
  #[512,1]
  var12657=tf.reshape(var12656, [512,1])
  #[512,65]
  var12658=tf.concat([var12655,var12657], axis=1)
  #[512,1]
  var12659=var12658[:,0:1]
  #[512,1]
  var12660=tf.reshape(var12659, [512,1])
  #[512,64]
  var12661=var12658[:,1:65]
  #[512,32,1,2]
  var12662=tf.reshape(var12661, [512,32,1,2])
  #[512,1,32]
  var12663=var12622[:,2:3]
  #[512,32]
  var12664=tf.reshape(var12663, [512,32])
  #[512,32]
  var12665=tf.cos(var12664)
  #[512,32,1]
  var12666=tf.reshape(var12665, [512,32,1])
  #[512,32]
  var12667=tf.sin(var12664)
  #[512,32]
  var12668=tf.negative(var12667)
  #[512,32,1]
  var12669=tf.reshape(var12668, [512,32,1])
  #[512,32,2]
  var12670=tf.concat([var12666,var12669], axis=2)
  #[512,32,1,2]
  var12671=tf.reshape(var12670, [512,32,1,2])
  #[512,32,1]
  var12672=tf.reshape(var12667, [512,32,1])
  #[512,32,2]
  var12673=tf.concat([var12672,var12666], axis=2)
  #[512,32,1,2]
  var12674=tf.reshape(var12673, [512,32,1,2])
  #[512,32,2,2]
  var12675=tf.concat([var12671,var12674], axis=2)
  #[512,32,1,2]
  var12676=tf.matmul(var12662, var12675)
  #[512,64]
  var12677=tf.reshape(var12676, [512,64])
  #[512,65]
  var12678=tf.concat([var12660,var12677], axis=1)
  #[512,64]
  var12679=var12678[:,0:64]
  #[512,32,1,2]
  var12680=tf.reshape(var12679, [512,32,1,2])
  #[512,1,32]
  var12681=var12622[:,1:2]
  #[512,32]
  var12682=tf.reshape(var12681, [512,32])
  #[512,32]
  var12683=tf.cos(var12682)
  #[512,32,1]
  var12684=tf.reshape(var12683, [512,32,1])
  #[512,32]
  var12685=tf.sin(var12682)
  #[512,32]
  var12686=tf.negative(var12685)
  #[512,32,1]
  var12687=tf.reshape(var12686, [512,32,1])
  #[512,32,2]
  var12688=tf.concat([var12684,var12687], axis=2)
  #[512,32,1,2]
  var12689=tf.reshape(var12688, [512,32,1,2])
  #[512,32,1]
  var12690=tf.reshape(var12685, [512,32,1])
  #[512,32,2]
  var12691=tf.concat([var12690,var12684], axis=2)
  #[512,32,1,2]
  var12692=tf.reshape(var12691, [512,32,1,2])
  #[512,32,2,2]
  var12693=tf.concat([var12689,var12692], axis=2)
  #[512,32,1,2]
  var12694=tf.matmul(var12680, var12693)
  #[512,64]
  var12695=tf.reshape(var12694, [512,64])
  #[512,1]
  var12696=var12678[:,64:65]
  #[512,1]
  var12697=tf.reshape(var12696, [512,1])
  #[512,65]
  var12698=tf.concat([var12695,var12697], axis=1)
  #[512,1]
  var12699=var12698[:,0:1]
  #[512,1]
  var12700=tf.reshape(var12699, [512,1])
  #[512,64]
  var12701=var12698[:,1:65]
  #[512,32,1,2]
  var12702=tf.reshape(var12701, [512,32,1,2])
  #[512,1,32]
  var12703=var12622[:,0:1]
  #[512,32]
  var12704=tf.reshape(var12703, [512,32])
  #[512,32]
  var12705=tf.cos(var12704)
  #[512,32,1]
  var12706=tf.reshape(var12705, [512,32,1])
  #[512,32]
  var12707=tf.sin(var12704)
  #[512,32]
  var12708=tf.negative(var12707)
  #[512,32,1]
  var12709=tf.reshape(var12708, [512,32,1])
  #[512,32,2]
  var12710=tf.concat([var12706,var12709], axis=2)
  #[512,32,1,2]
  var12711=tf.reshape(var12710, [512,32,1,2])
  #[512,32,1]
  var12712=tf.reshape(var12707, [512,32,1])
  #[512,32,2]
  var12713=tf.concat([var12712,var12706], axis=2)
  #[512,32,1,2]
  var12714=tf.reshape(var12713, [512,32,1,2])
  #[512,32,2,2]
  var12715=tf.concat([var12711,var12714], axis=2)
  #[512,32,1,2]
  var12716=tf.matmul(var12702, var12715)
  #[512,64]
  var12717=tf.reshape(var12716, [512,64])
  #[512,65]
  var12718=tf.concat([var12700,var12717], axis=1)
  #[512,65]
  var12719=tf.multiply(var12364, var12718)
  #[512,65]
  var12720=tf.reshape(var12719, [512,65])
  #[512,12]
  var12721=tf.matmul(var12720, var12494)
  #[512,12]
  var12722=tf.reshape(var12721, [512,12])
  #[512,12]
  var12723=tf.add(var12722, var12498)
  #[512,1,12]
  var12724=tf.reshape(var12723, [512,1,12])
  #[512,65]
  var12725=tf.multiply(var12368, var12718)
  #[512,1]
  var12726=var12725[:,0:1]
  #[512,1]
  var12727=tf.reshape(var12726, [512,1])
  #[512,64]
  var12728=var12725[:,1:65]
  #[512,32,1,2]
  var12729=tf.reshape(var12728, [512,32,1,2])
  #[512,1]
  var12730=var12390[:,3:4]
  #[512]
  var12731=tf.reshape(var12730, [512])
  #[512,160]
  var12732=tf.gather(params=var12389, indices=var12731, batch_dims=0, axis=0)
  #[512,160]
  var12733=tf.multiply(var12388, var12732)
  #[512,5,32]
  var12734=tf.reshape(var12733, [512,5,32])
  #[512,1,32]
  var12735=var12734[:,4:5]
  #[512,32]
  var12736=tf.reshape(var12735, [512,32])
  #[512,32]
  var12737=tf.cos(var12736)
  #[512,32,1]
  var12738=tf.reshape(var12737, [512,32,1])
  #[512,32]
  var12739=tf.sin(var12736)
  #[512,32]
  var12740=tf.negative(var12739)
  #[512,32,1]
  var12741=tf.reshape(var12740, [512,32,1])
  #[512,32,2]
  var12742=tf.concat([var12738,var12741], axis=2)
  #[512,32,1,2]
  var12743=tf.reshape(var12742, [512,32,1,2])
  #[512,32,1]
  var12744=tf.reshape(var12739, [512,32,1])
  #[512,32,2]
  var12745=tf.concat([var12744,var12738], axis=2)
  #[512,32,1,2]
  var12746=tf.reshape(var12745, [512,32,1,2])
  #[512,32,2,2]
  var12747=tf.concat([var12743,var12746], axis=2)
  #[512,32,1,2]
  var12748=tf.matmul(var12729, var12747)
  #[512,64]
  var12749=tf.reshape(var12748, [512,64])
  #[512,65]
  var12750=tf.concat([var12727,var12749], axis=1)
  #[512,64]
  var12751=var12750[:,0:64]
  #[512,32,1,2]
  var12752=tf.reshape(var12751, [512,32,1,2])
  #[512,1,32]
  var12753=var12734[:,3:4]
  #[512,32]
  var12754=tf.reshape(var12753, [512,32])
  #[512,32]
  var12755=tf.cos(var12754)
  #[512,32,1]
  var12756=tf.reshape(var12755, [512,32,1])
  #[512,32]
  var12757=tf.sin(var12754)
  #[512,32]
  var12758=tf.negative(var12757)
  #[512,32,1]
  var12759=tf.reshape(var12758, [512,32,1])
  #[512,32,2]
  var12760=tf.concat([var12756,var12759], axis=2)
  #[512,32,1,2]
  var12761=tf.reshape(var12760, [512,32,1,2])
  #[512,32,1]
  var12762=tf.reshape(var12757, [512,32,1])
  #[512,32,2]
  var12763=tf.concat([var12762,var12756], axis=2)
  #[512,32,1,2]
  var12764=tf.reshape(var12763, [512,32,1,2])
  #[512,32,2,2]
  var12765=tf.concat([var12761,var12764], axis=2)
  #[512,32,1,2]
  var12766=tf.matmul(var12752, var12765)
  #[512,64]
  var12767=tf.reshape(var12766, [512,64])
  #[512,1]
  var12768=var12750[:,64:65]
  #[512,1]
  var12769=tf.reshape(var12768, [512,1])
  #[512,65]
  var12770=tf.concat([var12767,var12769], axis=1)
  #[512,1]
  var12771=var12770[:,0:1]
  #[512,1]
  var12772=tf.reshape(var12771, [512,1])
  #[512,64]
  var12773=var12770[:,1:65]
  #[512,32,1,2]
  var12774=tf.reshape(var12773, [512,32,1,2])
  #[512,1,32]
  var12775=var12734[:,2:3]
  #[512,32]
  var12776=tf.reshape(var12775, [512,32])
  #[512,32]
  var12777=tf.cos(var12776)
  #[512,32,1]
  var12778=tf.reshape(var12777, [512,32,1])
  #[512,32]
  var12779=tf.sin(var12776)
  #[512,32]
  var12780=tf.negative(var12779)
  #[512,32,1]
  var12781=tf.reshape(var12780, [512,32,1])
  #[512,32,2]
  var12782=tf.concat([var12778,var12781], axis=2)
  #[512,32,1,2]
  var12783=tf.reshape(var12782, [512,32,1,2])
  #[512,32,1]
  var12784=tf.reshape(var12779, [512,32,1])
  #[512,32,2]
  var12785=tf.concat([var12784,var12778], axis=2)
  #[512,32,1,2]
  var12786=tf.reshape(var12785, [512,32,1,2])
  #[512,32,2,2]
  var12787=tf.concat([var12783,var12786], axis=2)
  #[512,32,1,2]
  var12788=tf.matmul(var12774, var12787)
  #[512,64]
  var12789=tf.reshape(var12788, [512,64])
  #[512,65]
  var12790=tf.concat([var12772,var12789], axis=1)
  #[512,64]
  var12791=var12790[:,0:64]
  #[512,32,1,2]
  var12792=tf.reshape(var12791, [512,32,1,2])
  #[512,1,32]
  var12793=var12734[:,1:2]
  #[512,32]
  var12794=tf.reshape(var12793, [512,32])
  #[512,32]
  var12795=tf.cos(var12794)
  #[512,32,1]
  var12796=tf.reshape(var12795, [512,32,1])
  #[512,32]
  var12797=tf.sin(var12794)
  #[512,32]
  var12798=tf.negative(var12797)
  #[512,32,1]
  var12799=tf.reshape(var12798, [512,32,1])
  #[512,32,2]
  var12800=tf.concat([var12796,var12799], axis=2)
  #[512,32,1,2]
  var12801=tf.reshape(var12800, [512,32,1,2])
  #[512,32,1]
  var12802=tf.reshape(var12797, [512,32,1])
  #[512,32,2]
  var12803=tf.concat([var12802,var12796], axis=2)
  #[512,32,1,2]
  var12804=tf.reshape(var12803, [512,32,1,2])
  #[512,32,2,2]
  var12805=tf.concat([var12801,var12804], axis=2)
  #[512,32,1,2]
  var12806=tf.matmul(var12792, var12805)
  #[512,64]
  var12807=tf.reshape(var12806, [512,64])
  #[512,1]
  var12808=var12790[:,64:65]
  #[512,1]
  var12809=tf.reshape(var12808, [512,1])
  #[512,65]
  var12810=tf.concat([var12807,var12809], axis=1)
  #[512,1]
  var12811=var12810[:,0:1]
  #[512,1]
  var12812=tf.reshape(var12811, [512,1])
  #[512,64]
  var12813=var12810[:,1:65]
  #[512,32,1,2]
  var12814=tf.reshape(var12813, [512,32,1,2])
  #[512,1,32]
  var12815=var12734[:,0:1]
  #[512,32]
  var12816=tf.reshape(var12815, [512,32])
  #[512,32]
  var12817=tf.cos(var12816)
  #[512,32,1]
  var12818=tf.reshape(var12817, [512,32,1])
  #[512,32]
  var12819=tf.sin(var12816)
  #[512,32]
  var12820=tf.negative(var12819)
  #[512,32,1]
  var12821=tf.reshape(var12820, [512,32,1])
  #[512,32,2]
  var12822=tf.concat([var12818,var12821], axis=2)
  #[512,32,1,2]
  var12823=tf.reshape(var12822, [512,32,1,2])
  #[512,32,1]
  var12824=tf.reshape(var12819, [512,32,1])
  #[512,32,2]
  var12825=tf.concat([var12824,var12818], axis=2)
  #[512,32,1,2]
  var12826=tf.reshape(var12825, [512,32,1,2])
  #[512,32,2,2]
  var12827=tf.concat([var12823,var12826], axis=2)
  #[512,32,1,2]
  var12828=tf.matmul(var12814, var12827)
  #[512,64]
  var12829=tf.reshape(var12828, [512,64])
  #[512,65]
  var12830=tf.concat([var12812,var12829], axis=1)
  #[512,65]
  var12831=tf.multiply(var12364, var12830)
  #[512,65]
  var12832=tf.reshape(var12831, [512,65])
  #[512,12]
  var12833=tf.matmul(var12832, var12494)
  #[512,12]
  var12834=tf.reshape(var12833, [512,12])
  #[512,12]
  var12835=tf.add(var12834, var12498)
  #[512,1,12]
  var12836=tf.reshape(var12835, [512,1,12])
  #[512,65]
  var12837=tf.multiply(var12368, var12830)
  #[512,1]
  var12838=var12837[:,0:1]
  #[512,1]
  var12839=tf.reshape(var12838, [512,1])
  #[512,64]
  var12840=var12837[:,1:65]
  #[512,32,1,2]
  var12841=tf.reshape(var12840, [512,32,1,2])
  #[512,1]
  var12842=var12390[:,4:5]
  #[512]
  var12843=tf.reshape(var12842, [512])
  #[512,160]
  var12844=tf.gather(params=var12389, indices=var12843, batch_dims=0, axis=0)
  #[512,160]
  var12845=tf.multiply(var12388, var12844)
  #[512,5,32]
  var12846=tf.reshape(var12845, [512,5,32])
  #[512,1,32]
  var12847=var12846[:,4:5]
  #[512,32]
  var12848=tf.reshape(var12847, [512,32])
  #[512,32]
  var12849=tf.cos(var12848)
  #[512,32,1]
  var12850=tf.reshape(var12849, [512,32,1])
  #[512,32]
  var12851=tf.sin(var12848)
  #[512,32]
  var12852=tf.negative(var12851)
  #[512,32,1]
  var12853=tf.reshape(var12852, [512,32,1])
  #[512,32,2]
  var12854=tf.concat([var12850,var12853], axis=2)
  #[512,32,1,2]
  var12855=tf.reshape(var12854, [512,32,1,2])
  #[512,32,1]
  var12856=tf.reshape(var12851, [512,32,1])
  #[512,32,2]
  var12857=tf.concat([var12856,var12850], axis=2)
  #[512,32,1,2]
  var12858=tf.reshape(var12857, [512,32,1,2])
  #[512,32,2,2]
  var12859=tf.concat([var12855,var12858], axis=2)
  #[512,32,1,2]
  var12860=tf.matmul(var12841, var12859)
  #[512,64]
  var12861=tf.reshape(var12860, [512,64])
  #[512,65]
  var12862=tf.concat([var12839,var12861], axis=1)
  #[512,64]
  var12863=var12862[:,0:64]
  #[512,32,1,2]
  var12864=tf.reshape(var12863, [512,32,1,2])
  #[512,1,32]
  var12865=var12846[:,3:4]
  #[512,32]
  var12866=tf.reshape(var12865, [512,32])
  #[512,32]
  var12867=tf.cos(var12866)
  #[512,32,1]
  var12868=tf.reshape(var12867, [512,32,1])
  #[512,32]
  var12869=tf.sin(var12866)
  #[512,32]
  var12870=tf.negative(var12869)
  #[512,32,1]
  var12871=tf.reshape(var12870, [512,32,1])
  #[512,32,2]
  var12872=tf.concat([var12868,var12871], axis=2)
  #[512,32,1,2]
  var12873=tf.reshape(var12872, [512,32,1,2])
  #[512,32,1]
  var12874=tf.reshape(var12869, [512,32,1])
  #[512,32,2]
  var12875=tf.concat([var12874,var12868], axis=2)
  #[512,32,1,2]
  var12876=tf.reshape(var12875, [512,32,1,2])
  #[512,32,2,2]
  var12877=tf.concat([var12873,var12876], axis=2)
  #[512,32,1,2]
  var12878=tf.matmul(var12864, var12877)
  #[512,64]
  var12879=tf.reshape(var12878, [512,64])
  #[512,1]
  var12880=var12862[:,64:65]
  #[512,1]
  var12881=tf.reshape(var12880, [512,1])
  #[512,65]
  var12882=tf.concat([var12879,var12881], axis=1)
  #[512,1]
  var12883=var12882[:,0:1]
  #[512,1]
  var12884=tf.reshape(var12883, [512,1])
  #[512,64]
  var12885=var12882[:,1:65]
  #[512,32,1,2]
  var12886=tf.reshape(var12885, [512,32,1,2])
  #[512,1,32]
  var12887=var12846[:,2:3]
  #[512,32]
  var12888=tf.reshape(var12887, [512,32])
  #[512,32]
  var12889=tf.cos(var12888)
  #[512,32,1]
  var12890=tf.reshape(var12889, [512,32,1])
  #[512,32]
  var12891=tf.sin(var12888)
  #[512,32]
  var12892=tf.negative(var12891)
  #[512,32,1]
  var12893=tf.reshape(var12892, [512,32,1])
  #[512,32,2]
  var12894=tf.concat([var12890,var12893], axis=2)
  #[512,32,1,2]
  var12895=tf.reshape(var12894, [512,32,1,2])
  #[512,32,1]
  var12896=tf.reshape(var12891, [512,32,1])
  #[512,32,2]
  var12897=tf.concat([var12896,var12890], axis=2)
  #[512,32,1,2]
  var12898=tf.reshape(var12897, [512,32,1,2])
  #[512,32,2,2]
  var12899=tf.concat([var12895,var12898], axis=2)
  #[512,32,1,2]
  var12900=tf.matmul(var12886, var12899)
  #[512,64]
  var12901=tf.reshape(var12900, [512,64])
  #[512,65]
  var12902=tf.concat([var12884,var12901], axis=1)
  #[512,64]
  var12903=var12902[:,0:64]
  #[512,32,1,2]
  var12904=tf.reshape(var12903, [512,32,1,2])
  #[512,1,32]
  var12905=var12846[:,1:2]
  #[512,32]
  var12906=tf.reshape(var12905, [512,32])
  #[512,32]
  var12907=tf.cos(var12906)
  #[512,32,1]
  var12908=tf.reshape(var12907, [512,32,1])
  #[512,32]
  var12909=tf.sin(var12906)
  #[512,32]
  var12910=tf.negative(var12909)
  #[512,32,1]
  var12911=tf.reshape(var12910, [512,32,1])
  #[512,32,2]
  var12912=tf.concat([var12908,var12911], axis=2)
  #[512,32,1,2]
  var12913=tf.reshape(var12912, [512,32,1,2])
  #[512,32,1]
  var12914=tf.reshape(var12909, [512,32,1])
  #[512,32,2]
  var12915=tf.concat([var12914,var12908], axis=2)
  #[512,32,1,2]
  var12916=tf.reshape(var12915, [512,32,1,2])
  #[512,32,2,2]
  var12917=tf.concat([var12913,var12916], axis=2)
  #[512,32,1,2]
  var12918=tf.matmul(var12904, var12917)
  #[512,64]
  var12919=tf.reshape(var12918, [512,64])
  #[512,1]
  var12920=var12902[:,64:65]
  #[512,1]
  var12921=tf.reshape(var12920, [512,1])
  #[512,65]
  var12922=tf.concat([var12919,var12921], axis=1)
  #[512,1]
  var12923=var12922[:,0:1]
  #[512,1]
  var12924=tf.reshape(var12923, [512,1])
  #[512,64]
  var12925=var12922[:,1:65]
  #[512,32,1,2]
  var12926=tf.reshape(var12925, [512,32,1,2])
  #[512,1,32]
  var12927=var12846[:,0:1]
  #[512,32]
  var12928=tf.reshape(var12927, [512,32])
  #[512,32]
  var12929=tf.cos(var12928)
  #[512,32,1]
  var12930=tf.reshape(var12929, [512,32,1])
  #[512,32]
  var12931=tf.sin(var12928)
  #[512,32]
  var12932=tf.negative(var12931)
  #[512,32,1]
  var12933=tf.reshape(var12932, [512,32,1])
  #[512,32,2]
  var12934=tf.concat([var12930,var12933], axis=2)
  #[512,32,1,2]
  var12935=tf.reshape(var12934, [512,32,1,2])
  #[512,32,1]
  var12936=tf.reshape(var12931, [512,32,1])
  #[512,32,2]
  var12937=tf.concat([var12936,var12930], axis=2)
  #[512,32,1,2]
  var12938=tf.reshape(var12937, [512,32,1,2])
  #[512,32,2,2]
  var12939=tf.concat([var12935,var12938], axis=2)
  #[512,32,1,2]
  var12940=tf.matmul(var12926, var12939)
  #[512,64]
  var12941=tf.reshape(var12940, [512,64])
  #[512,65]
  var12942=tf.concat([var12924,var12941], axis=1)
  #[512,65]
  var12943=tf.multiply(var12364, var12942)
  #[512,65]
  var12944=tf.reshape(var12943, [512,65])
  #[512,12]
  var12945=tf.matmul(var12944, var12494)
  #[512,12]
  var12946=tf.reshape(var12945, [512,12])
  #[512,12]
  var12947=tf.add(var12946, var12498)
  #[512,1,12]
  var12948=tf.reshape(var12947, [512,1,12])
  #[512,65]
  var12949=tf.multiply(var12368, var12942)
  #[512,1]
  var12950=var12949[:,0:1]
  #[512,1]
  var12951=tf.reshape(var12950, [512,1])
  #[512,64]
  var12952=var12949[:,1:65]
  #[512,32,1,2]
  var12953=tf.reshape(var12952, [512,32,1,2])
  #[512,1]
  var12954=var12390[:,5:6]
  #[512]
  var12955=tf.reshape(var12954, [512])
  #[512,160]
  var12956=tf.gather(params=var12389, indices=var12955, batch_dims=0, axis=0)
  #[512,160]
  var12957=tf.multiply(var12388, var12956)
  #[512,5,32]
  var12958=tf.reshape(var12957, [512,5,32])
  #[512,1,32]
  var12959=var12958[:,4:5]
  #[512,32]
  var12960=tf.reshape(var12959, [512,32])
  #[512,32]
  var12961=tf.cos(var12960)
  #[512,32,1]
  var12962=tf.reshape(var12961, [512,32,1])
  #[512,32]
  var12963=tf.sin(var12960)
  #[512,32]
  var12964=tf.negative(var12963)
  #[512,32,1]
  var12965=tf.reshape(var12964, [512,32,1])
  #[512,32,2]
  var12966=tf.concat([var12962,var12965], axis=2)
  #[512,32,1,2]
  var12967=tf.reshape(var12966, [512,32,1,2])
  #[512,32,1]
  var12968=tf.reshape(var12963, [512,32,1])
  #[512,32,2]
  var12969=tf.concat([var12968,var12962], axis=2)
  #[512,32,1,2]
  var12970=tf.reshape(var12969, [512,32,1,2])
  #[512,32,2,2]
  var12971=tf.concat([var12967,var12970], axis=2)
  #[512,32,1,2]
  var12972=tf.matmul(var12953, var12971)
  #[512,64]
  var12973=tf.reshape(var12972, [512,64])
  #[512,65]
  var12974=tf.concat([var12951,var12973], axis=1)
  #[512,64]
  var12975=var12974[:,0:64]
  #[512,32,1,2]
  var12976=tf.reshape(var12975, [512,32,1,2])
  #[512,1,32]
  var12977=var12958[:,3:4]
  #[512,32]
  var12978=tf.reshape(var12977, [512,32])
  #[512,32]
  var12979=tf.cos(var12978)
  #[512,32,1]
  var12980=tf.reshape(var12979, [512,32,1])
  #[512,32]
  var12981=tf.sin(var12978)
  #[512,32]
  var12982=tf.negative(var12981)
  #[512,32,1]
  var12983=tf.reshape(var12982, [512,32,1])
  #[512,32,2]
  var12984=tf.concat([var12980,var12983], axis=2)
  #[512,32,1,2]
  var12985=tf.reshape(var12984, [512,32,1,2])
  #[512,32,1]
  var12986=tf.reshape(var12981, [512,32,1])
  #[512,32,2]
  var12987=tf.concat([var12986,var12980], axis=2)
  #[512,32,1,2]
  var12988=tf.reshape(var12987, [512,32,1,2])
  #[512,32,2,2]
  var12989=tf.concat([var12985,var12988], axis=2)
  #[512,32,1,2]
  var12990=tf.matmul(var12976, var12989)
  #[512,64]
  var12991=tf.reshape(var12990, [512,64])
  #[512,1]
  var12992=var12974[:,64:65]
  #[512,1]
  var12993=tf.reshape(var12992, [512,1])
  #[512,65]
  var12994=tf.concat([var12991,var12993], axis=1)
  #[512,1]
  var12995=var12994[:,0:1]
  #[512,1]
  var12996=tf.reshape(var12995, [512,1])
  #[512,64]
  var12997=var12994[:,1:65]
  #[512,32,1,2]
  var12998=tf.reshape(var12997, [512,32,1,2])
  #[512,1,32]
  var12999=var12958[:,2:3]
  #[512,32]
  var13000=tf.reshape(var12999, [512,32])
  #[512,32]
  var13001=tf.cos(var13000)
  #[512,32,1]
  var13002=tf.reshape(var13001, [512,32,1])
  #[512,32]
  var13003=tf.sin(var13000)
  #[512,32]
  var13004=tf.negative(var13003)
  #[512,32,1]
  var13005=tf.reshape(var13004, [512,32,1])
  #[512,32,2]
  var13006=tf.concat([var13002,var13005], axis=2)
  #[512,32,1,2]
  var13007=tf.reshape(var13006, [512,32,1,2])
  #[512,32,1]
  var13008=tf.reshape(var13003, [512,32,1])
  #[512,32,2]
  var13009=tf.concat([var13008,var13002], axis=2)
  #[512,32,1,2]
  var13010=tf.reshape(var13009, [512,32,1,2])
  #[512,32,2,2]
  var13011=tf.concat([var13007,var13010], axis=2)
  #[512,32,1,2]
  var13012=tf.matmul(var12998, var13011)
  #[512,64]
  var13013=tf.reshape(var13012, [512,64])
  #[512,65]
  var13014=tf.concat([var12996,var13013], axis=1)
  #[512,64]
  var13015=var13014[:,0:64]
  #[512,32,1,2]
  var13016=tf.reshape(var13015, [512,32,1,2])
  #[512,1,32]
  var13017=var12958[:,1:2]
  #[512,32]
  var13018=tf.reshape(var13017, [512,32])
  #[512,32]
  var13019=tf.cos(var13018)
  #[512,32,1]
  var13020=tf.reshape(var13019, [512,32,1])
  #[512,32]
  var13021=tf.sin(var13018)
  #[512,32]
  var13022=tf.negative(var13021)
  #[512,32,1]
  var13023=tf.reshape(var13022, [512,32,1])
  #[512,32,2]
  var13024=tf.concat([var13020,var13023], axis=2)
  #[512,32,1,2]
  var13025=tf.reshape(var13024, [512,32,1,2])
  #[512,32,1]
  var13026=tf.reshape(var13021, [512,32,1])
  #[512,32,2]
  var13027=tf.concat([var13026,var13020], axis=2)
  #[512,32,1,2]
  var13028=tf.reshape(var13027, [512,32,1,2])
  #[512,32,2,2]
  var13029=tf.concat([var13025,var13028], axis=2)
  #[512,32,1,2]
  var13030=tf.matmul(var13016, var13029)
  #[512,64]
  var13031=tf.reshape(var13030, [512,64])
  #[512,1]
  var13032=var13014[:,64:65]
  #[512,1]
  var13033=tf.reshape(var13032, [512,1])
  #[512,65]
  var13034=tf.concat([var13031,var13033], axis=1)
  #[512,1]
  var13035=var13034[:,0:1]
  #[512,1]
  var13036=tf.reshape(var13035, [512,1])
  #[512,64]
  var13037=var13034[:,1:65]
  #[512,32,1,2]
  var13038=tf.reshape(var13037, [512,32,1,2])
  #[512,1,32]
  var13039=var12958[:,0:1]
  #[512,32]
  var13040=tf.reshape(var13039, [512,32])
  #[512,32]
  var13041=tf.cos(var13040)
  #[512,32,1]
  var13042=tf.reshape(var13041, [512,32,1])
  #[512,32]
  var13043=tf.sin(var13040)
  #[512,32]
  var13044=tf.negative(var13043)
  #[512,32,1]
  var13045=tf.reshape(var13044, [512,32,1])
  #[512,32,2]
  var13046=tf.concat([var13042,var13045], axis=2)
  #[512,32,1,2]
  var13047=tf.reshape(var13046, [512,32,1,2])
  #[512,32,1]
  var13048=tf.reshape(var13043, [512,32,1])
  #[512,32,2]
  var13049=tf.concat([var13048,var13042], axis=2)
  #[512,32,1,2]
  var13050=tf.reshape(var13049, [512,32,1,2])
  #[512,32,2,2]
  var13051=tf.concat([var13047,var13050], axis=2)
  #[512,32,1,2]
  var13052=tf.matmul(var13038, var13051)
  #[512,64]
  var13053=tf.reshape(var13052, [512,64])
  #[512,65]
  var13054=tf.concat([var13036,var13053], axis=1)
  #[512,65]
  var13055=tf.multiply(var12364, var13054)
  #[512,65]
  var13056=tf.reshape(var13055, [512,65])
  #[512,12]
  var13057=tf.matmul(var13056, var12494)
  #[512,12]
  var13058=tf.reshape(var13057, [512,12])
  #[512,12]
  var13059=tf.add(var13058, var12498)
  #[512,1,12]
  var13060=tf.reshape(var13059, [512,1,12])
  #[512,65]
  var13061=tf.multiply(var12368, var13054)
  #[512,1]
  var13062=var13061[:,0:1]
  #[512,1]
  var13063=tf.reshape(var13062, [512,1])
  #[512,64]
  var13064=var13061[:,1:65]
  #[512,32,1,2]
  var13065=tf.reshape(var13064, [512,32,1,2])
  #[512,1]
  var13066=var12390[:,6:7]
  #[512]
  var13067=tf.reshape(var13066, [512])
  #[512,160]
  var13068=tf.gather(params=var12389, indices=var13067, batch_dims=0, axis=0)
  #[512,160]
  var13069=tf.multiply(var12388, var13068)
  #[512,5,32]
  var13070=tf.reshape(var13069, [512,5,32])
  #[512,1,32]
  var13071=var13070[:,4:5]
  #[512,32]
  var13072=tf.reshape(var13071, [512,32])
  #[512,32]
  var13073=tf.cos(var13072)
  #[512,32,1]
  var13074=tf.reshape(var13073, [512,32,1])
  #[512,32]
  var13075=tf.sin(var13072)
  #[512,32]
  var13076=tf.negative(var13075)
  #[512,32,1]
  var13077=tf.reshape(var13076, [512,32,1])
  #[512,32,2]
  var13078=tf.concat([var13074,var13077], axis=2)
  #[512,32,1,2]
  var13079=tf.reshape(var13078, [512,32,1,2])
  #[512,32,1]
  var13080=tf.reshape(var13075, [512,32,1])
  #[512,32,2]
  var13081=tf.concat([var13080,var13074], axis=2)
  #[512,32,1,2]
  var13082=tf.reshape(var13081, [512,32,1,2])
  #[512,32,2,2]
  var13083=tf.concat([var13079,var13082], axis=2)
  #[512,32,1,2]
  var13084=tf.matmul(var13065, var13083)
  #[512,64]
  var13085=tf.reshape(var13084, [512,64])
  #[512,65]
  var13086=tf.concat([var13063,var13085], axis=1)
  #[512,64]
  var13087=var13086[:,0:64]
  #[512,32,1,2]
  var13088=tf.reshape(var13087, [512,32,1,2])
  #[512,1,32]
  var13089=var13070[:,3:4]
  #[512,32]
  var13090=tf.reshape(var13089, [512,32])
  #[512,32]
  var13091=tf.cos(var13090)
  #[512,32,1]
  var13092=tf.reshape(var13091, [512,32,1])
  #[512,32]
  var13093=tf.sin(var13090)
  #[512,32]
  var13094=tf.negative(var13093)
  #[512,32,1]
  var13095=tf.reshape(var13094, [512,32,1])
  #[512,32,2]
  var13096=tf.concat([var13092,var13095], axis=2)
  #[512,32,1,2]
  var13097=tf.reshape(var13096, [512,32,1,2])
  #[512,32,1]
  var13098=tf.reshape(var13093, [512,32,1])
  #[512,32,2]
  var13099=tf.concat([var13098,var13092], axis=2)
  #[512,32,1,2]
  var13100=tf.reshape(var13099, [512,32,1,2])
  #[512,32,2,2]
  var13101=tf.concat([var13097,var13100], axis=2)
  #[512,32,1,2]
  var13102=tf.matmul(var13088, var13101)
  #[512,64]
  var13103=tf.reshape(var13102, [512,64])
  #[512,1]
  var13104=var13086[:,64:65]
  #[512,1]
  var13105=tf.reshape(var13104, [512,1])
  #[512,65]
  var13106=tf.concat([var13103,var13105], axis=1)
  #[512,1]
  var13107=var13106[:,0:1]
  #[512,1]
  var13108=tf.reshape(var13107, [512,1])
  #[512,64]
  var13109=var13106[:,1:65]
  #[512,32,1,2]
  var13110=tf.reshape(var13109, [512,32,1,2])
  #[512,1,32]
  var13111=var13070[:,2:3]
  #[512,32]
  var13112=tf.reshape(var13111, [512,32])
  #[512,32]
  var13113=tf.cos(var13112)
  #[512,32,1]
  var13114=tf.reshape(var13113, [512,32,1])
  #[512,32]
  var13115=tf.sin(var13112)
  #[512,32]
  var13116=tf.negative(var13115)
  #[512,32,1]
  var13117=tf.reshape(var13116, [512,32,1])
  #[512,32,2]
  var13118=tf.concat([var13114,var13117], axis=2)
  #[512,32,1,2]
  var13119=tf.reshape(var13118, [512,32,1,2])
  #[512,32,1]
  var13120=tf.reshape(var13115, [512,32,1])
  #[512,32,2]
  var13121=tf.concat([var13120,var13114], axis=2)
  #[512,32,1,2]
  var13122=tf.reshape(var13121, [512,32,1,2])
  #[512,32,2,2]
  var13123=tf.concat([var13119,var13122], axis=2)
  #[512,32,1,2]
  var13124=tf.matmul(var13110, var13123)
  #[512,64]
  var13125=tf.reshape(var13124, [512,64])
  #[512,65]
  var13126=tf.concat([var13108,var13125], axis=1)
  #[512,64]
  var13127=var13126[:,0:64]
  #[512,32,1,2]
  var13128=tf.reshape(var13127, [512,32,1,2])
  #[512,1,32]
  var13129=var13070[:,1:2]
  #[512,32]
  var13130=tf.reshape(var13129, [512,32])
  #[512,32]
  var13131=tf.cos(var13130)
  #[512,32,1]
  var13132=tf.reshape(var13131, [512,32,1])
  #[512,32]
  var13133=tf.sin(var13130)
  #[512,32]
  var13134=tf.negative(var13133)
  #[512,32,1]
  var13135=tf.reshape(var13134, [512,32,1])
  #[512,32,2]
  var13136=tf.concat([var13132,var13135], axis=2)
  #[512,32,1,2]
  var13137=tf.reshape(var13136, [512,32,1,2])
  #[512,32,1]
  var13138=tf.reshape(var13133, [512,32,1])
  #[512,32,2]
  var13139=tf.concat([var13138,var13132], axis=2)
  #[512,32,1,2]
  var13140=tf.reshape(var13139, [512,32,1,2])
  #[512,32,2,2]
  var13141=tf.concat([var13137,var13140], axis=2)
  #[512,32,1,2]
  var13142=tf.matmul(var13128, var13141)
  #[512,64]
  var13143=tf.reshape(var13142, [512,64])
  #[512,1]
  var13144=var13126[:,64:65]
  #[512,1]
  var13145=tf.reshape(var13144, [512,1])
  #[512,65]
  var13146=tf.concat([var13143,var13145], axis=1)
  #[512,1]
  var13147=var13146[:,0:1]
  #[512,1]
  var13148=tf.reshape(var13147, [512,1])
  #[512,64]
  var13149=var13146[:,1:65]
  #[512,32,1,2]
  var13150=tf.reshape(var13149, [512,32,1,2])
  #[512,1,32]
  var13151=var13070[:,0:1]
  #[512,32]
  var13152=tf.reshape(var13151, [512,32])
  #[512,32]
  var13153=tf.cos(var13152)
  #[512,32,1]
  var13154=tf.reshape(var13153, [512,32,1])
  #[512,32]
  var13155=tf.sin(var13152)
  #[512,32]
  var13156=tf.negative(var13155)
  #[512,32,1]
  var13157=tf.reshape(var13156, [512,32,1])
  #[512,32,2]
  var13158=tf.concat([var13154,var13157], axis=2)
  #[512,32,1,2]
  var13159=tf.reshape(var13158, [512,32,1,2])
  #[512,32,1]
  var13160=tf.reshape(var13155, [512,32,1])
  #[512,32,2]
  var13161=tf.concat([var13160,var13154], axis=2)
  #[512,32,1,2]
  var13162=tf.reshape(var13161, [512,32,1,2])
  #[512,32,2,2]
  var13163=tf.concat([var13159,var13162], axis=2)
  #[512,32,1,2]
  var13164=tf.matmul(var13150, var13163)
  #[512,64]
  var13165=tf.reshape(var13164, [512,64])
  #[512,65]
  var13166=tf.concat([var13148,var13165], axis=1)
  #[512,65]
  var13167=tf.multiply(var12364, var13166)
  #[512,65]
  var13168=tf.reshape(var13167, [512,65])
  #[512,12]
  var13169=tf.matmul(var13168, var12494)
  #[512,12]
  var13170=tf.reshape(var13169, [512,12])
  #[512,12]
  var13171=tf.add(var13170, var12498)
  #[512,1,12]
  var13172=tf.reshape(var13171, [512,1,12])
  #[512,65]
  var13173=tf.multiply(var12368, var13166)
  #[512,1]
  var13174=var13173[:,0:1]
  #[512,1]
  var13175=tf.reshape(var13174, [512,1])
  #[512,64]
  var13176=var13173[:,1:65]
  #[512,32,1,2]
  var13177=tf.reshape(var13176, [512,32,1,2])
  #[512,1]
  var13178=var12390[:,7:8]
  #[512]
  var13179=tf.reshape(var13178, [512])
  #[512,160]
  var13180=tf.gather(params=var12389, indices=var13179, batch_dims=0, axis=0)
  #[512,160]
  var13181=tf.multiply(var12388, var13180)
  #[512,5,32]
  var13182=tf.reshape(var13181, [512,5,32])
  #[512,1,32]
  var13183=var13182[:,4:5]
  #[512,32]
  var13184=tf.reshape(var13183, [512,32])
  #[512,32]
  var13185=tf.cos(var13184)
  #[512,32,1]
  var13186=tf.reshape(var13185, [512,32,1])
  #[512,32]
  var13187=tf.sin(var13184)
  #[512,32]
  var13188=tf.negative(var13187)
  #[512,32,1]
  var13189=tf.reshape(var13188, [512,32,1])
  #[512,32,2]
  var13190=tf.concat([var13186,var13189], axis=2)
  #[512,32,1,2]
  var13191=tf.reshape(var13190, [512,32,1,2])
  #[512,32,1]
  var13192=tf.reshape(var13187, [512,32,1])
  #[512,32,2]
  var13193=tf.concat([var13192,var13186], axis=2)
  #[512,32,1,2]
  var13194=tf.reshape(var13193, [512,32,1,2])
  #[512,32,2,2]
  var13195=tf.concat([var13191,var13194], axis=2)
  #[512,32,1,2]
  var13196=tf.matmul(var13177, var13195)
  #[512,64]
  var13197=tf.reshape(var13196, [512,64])
  #[512,65]
  var13198=tf.concat([var13175,var13197], axis=1)
  #[512,64]
  var13199=var13198[:,0:64]
  #[512,32,1,2]
  var13200=tf.reshape(var13199, [512,32,1,2])
  #[512,1,32]
  var13201=var13182[:,3:4]
  #[512,32]
  var13202=tf.reshape(var13201, [512,32])
  #[512,32]
  var13203=tf.cos(var13202)
  #[512,32,1]
  var13204=tf.reshape(var13203, [512,32,1])
  #[512,32]
  var13205=tf.sin(var13202)
  #[512,32]
  var13206=tf.negative(var13205)
  #[512,32,1]
  var13207=tf.reshape(var13206, [512,32,1])
  #[512,32,2]
  var13208=tf.concat([var13204,var13207], axis=2)
  #[512,32,1,2]
  var13209=tf.reshape(var13208, [512,32,1,2])
  #[512,32,1]
  var13210=tf.reshape(var13205, [512,32,1])
  #[512,32,2]
  var13211=tf.concat([var13210,var13204], axis=2)
  #[512,32,1,2]
  var13212=tf.reshape(var13211, [512,32,1,2])
  #[512,32,2,2]
  var13213=tf.concat([var13209,var13212], axis=2)
  #[512,32,1,2]
  var13214=tf.matmul(var13200, var13213)
  #[512,64]
  var13215=tf.reshape(var13214, [512,64])
  #[512,1]
  var13216=var13198[:,64:65]
  #[512,1]
  var13217=tf.reshape(var13216, [512,1])
  #[512,65]
  var13218=tf.concat([var13215,var13217], axis=1)
  #[512,1]
  var13219=var13218[:,0:1]
  #[512,1]
  var13220=tf.reshape(var13219, [512,1])
  #[512,64]
  var13221=var13218[:,1:65]
  #[512,32,1,2]
  var13222=tf.reshape(var13221, [512,32,1,2])
  #[512,1,32]
  var13223=var13182[:,2:3]
  #[512,32]
  var13224=tf.reshape(var13223, [512,32])
  #[512,32]
  var13225=tf.cos(var13224)
  #[512,32,1]
  var13226=tf.reshape(var13225, [512,32,1])
  #[512,32]
  var13227=tf.sin(var13224)
  #[512,32]
  var13228=tf.negative(var13227)
  #[512,32,1]
  var13229=tf.reshape(var13228, [512,32,1])
  #[512,32,2]
  var13230=tf.concat([var13226,var13229], axis=2)
  #[512,32,1,2]
  var13231=tf.reshape(var13230, [512,32,1,2])
  #[512,32,1]
  var13232=tf.reshape(var13227, [512,32,1])
  #[512,32,2]
  var13233=tf.concat([var13232,var13226], axis=2)
  #[512,32,1,2]
  var13234=tf.reshape(var13233, [512,32,1,2])
  #[512,32,2,2]
  var13235=tf.concat([var13231,var13234], axis=2)
  #[512,32,1,2]
  var13236=tf.matmul(var13222, var13235)
  #[512,64]
  var13237=tf.reshape(var13236, [512,64])
  #[512,65]
  var13238=tf.concat([var13220,var13237], axis=1)
  #[512,64]
  var13239=var13238[:,0:64]
  #[512,32,1,2]
  var13240=tf.reshape(var13239, [512,32,1,2])
  #[512,1,32]
  var13241=var13182[:,1:2]
  #[512,32]
  var13242=tf.reshape(var13241, [512,32])
  #[512,32]
  var13243=tf.cos(var13242)
  #[512,32,1]
  var13244=tf.reshape(var13243, [512,32,1])
  #[512,32]
  var13245=tf.sin(var13242)
  #[512,32]
  var13246=tf.negative(var13245)
  #[512,32,1]
  var13247=tf.reshape(var13246, [512,32,1])
  #[512,32,2]
  var13248=tf.concat([var13244,var13247], axis=2)
  #[512,32,1,2]
  var13249=tf.reshape(var13248, [512,32,1,2])
  #[512,32,1]
  var13250=tf.reshape(var13245, [512,32,1])
  #[512,32,2]
  var13251=tf.concat([var13250,var13244], axis=2)
  #[512,32,1,2]
  var13252=tf.reshape(var13251, [512,32,1,2])
  #[512,32,2,2]
  var13253=tf.concat([var13249,var13252], axis=2)
  #[512,32,1,2]
  var13254=tf.matmul(var13240, var13253)
  #[512,64]
  var13255=tf.reshape(var13254, [512,64])
  #[512,1]
  var13256=var13238[:,64:65]
  #[512,1]
  var13257=tf.reshape(var13256, [512,1])
  #[512,65]
  var13258=tf.concat([var13255,var13257], axis=1)
  #[512,1]
  var13259=var13258[:,0:1]
  #[512,1]
  var13260=tf.reshape(var13259, [512,1])
  #[512,64]
  var13261=var13258[:,1:65]
  #[512,32,1,2]
  var13262=tf.reshape(var13261, [512,32,1,2])
  #[512,1,32]
  var13263=var13182[:,0:1]
  #[512,32]
  var13264=tf.reshape(var13263, [512,32])
  #[512,32]
  var13265=tf.cos(var13264)
  #[512,32,1]
  var13266=tf.reshape(var13265, [512,32,1])
  #[512,32]
  var13267=tf.sin(var13264)
  #[512,32]
  var13268=tf.negative(var13267)
  #[512,32,1]
  var13269=tf.reshape(var13268, [512,32,1])
  #[512,32,2]
  var13270=tf.concat([var13266,var13269], axis=2)
  #[512,32,1,2]
  var13271=tf.reshape(var13270, [512,32,1,2])
  #[512,32,1]
  var13272=tf.reshape(var13267, [512,32,1])
  #[512,32,2]
  var13273=tf.concat([var13272,var13266], axis=2)
  #[512,32,1,2]
  var13274=tf.reshape(var13273, [512,32,1,2])
  #[512,32,2,2]
  var13275=tf.concat([var13271,var13274], axis=2)
  #[512,32,1,2]
  var13276=tf.matmul(var13262, var13275)
  #[512,64]
  var13277=tf.reshape(var13276, [512,64])
  #[512,65]
  var13278=tf.concat([var13260,var13277], axis=1)
  #[512,65]
  var13279=tf.multiply(var12364, var13278)
  #[512,65]
  var13280=tf.reshape(var13279, [512,65])
  #[512,12]
  var13281=tf.matmul(var13280, var12494)
  #[512,12]
  var13282=tf.reshape(var13281, [512,12])
  #[512,12]
  var13283=tf.add(var13282, var12498)
  #[512,1,12]
  var13284=tf.reshape(var13283, [512,1,12])
  #[512,65]
  var13285=tf.multiply(var12368, var13278)
  #[512,1]
  var13286=var13285[:,0:1]
  #[512,1]
  var13287=tf.reshape(var13286, [512,1])
  #[512,64]
  var13288=var13285[:,1:65]
  #[512,32,1,2]
  var13289=tf.reshape(var13288, [512,32,1,2])
  #[512,1]
  var13290=var12390[:,8:9]
  #[512]
  var13291=tf.reshape(var13290, [512])
  #[512,160]
  var13292=tf.gather(params=var12389, indices=var13291, batch_dims=0, axis=0)
  #[512,160]
  var13293=tf.multiply(var12388, var13292)
  #[512,5,32]
  var13294=tf.reshape(var13293, [512,5,32])
  #[512,1,32]
  var13295=var13294[:,4:5]
  #[512,32]
  var13296=tf.reshape(var13295, [512,32])
  #[512,32]
  var13297=tf.cos(var13296)
  #[512,32,1]
  var13298=tf.reshape(var13297, [512,32,1])
  #[512,32]
  var13299=tf.sin(var13296)
  #[512,32]
  var13300=tf.negative(var13299)
  #[512,32,1]
  var13301=tf.reshape(var13300, [512,32,1])
  #[512,32,2]
  var13302=tf.concat([var13298,var13301], axis=2)
  #[512,32,1,2]
  var13303=tf.reshape(var13302, [512,32,1,2])
  #[512,32,1]
  var13304=tf.reshape(var13299, [512,32,1])
  #[512,32,2]
  var13305=tf.concat([var13304,var13298], axis=2)
  #[512,32,1,2]
  var13306=tf.reshape(var13305, [512,32,1,2])
  #[512,32,2,2]
  var13307=tf.concat([var13303,var13306], axis=2)
  #[512,32,1,2]
  var13308=tf.matmul(var13289, var13307)
  #[512,64]
  var13309=tf.reshape(var13308, [512,64])
  #[512,65]
  var13310=tf.concat([var13287,var13309], axis=1)
  #[512,64]
  var13311=var13310[:,0:64]
  #[512,32,1,2]
  var13312=tf.reshape(var13311, [512,32,1,2])
  #[512,1,32]
  var13313=var13294[:,3:4]
  #[512,32]
  var13314=tf.reshape(var13313, [512,32])
  #[512,32]
  var13315=tf.cos(var13314)
  #[512,32,1]
  var13316=tf.reshape(var13315, [512,32,1])
  #[512,32]
  var13317=tf.sin(var13314)
  #[512,32]
  var13318=tf.negative(var13317)
  #[512,32,1]
  var13319=tf.reshape(var13318, [512,32,1])
  #[512,32,2]
  var13320=tf.concat([var13316,var13319], axis=2)
  #[512,32,1,2]
  var13321=tf.reshape(var13320, [512,32,1,2])
  #[512,32,1]
  var13322=tf.reshape(var13317, [512,32,1])
  #[512,32,2]
  var13323=tf.concat([var13322,var13316], axis=2)
  #[512,32,1,2]
  var13324=tf.reshape(var13323, [512,32,1,2])
  #[512,32,2,2]
  var13325=tf.concat([var13321,var13324], axis=2)
  #[512,32,1,2]
  var13326=tf.matmul(var13312, var13325)
  #[512,64]
  var13327=tf.reshape(var13326, [512,64])
  #[512,1]
  var13328=var13310[:,64:65]
  #[512,1]
  var13329=tf.reshape(var13328, [512,1])
  #[512,65]
  var13330=tf.concat([var13327,var13329], axis=1)
  #[512,1]
  var13331=var13330[:,0:1]
  #[512,1]
  var13332=tf.reshape(var13331, [512,1])
  #[512,64]
  var13333=var13330[:,1:65]
  #[512,32,1,2]
  var13334=tf.reshape(var13333, [512,32,1,2])
  #[512,1,32]
  var13335=var13294[:,2:3]
  #[512,32]
  var13336=tf.reshape(var13335, [512,32])
  #[512,32]
  var13337=tf.cos(var13336)
  #[512,32,1]
  var13338=tf.reshape(var13337, [512,32,1])
  #[512,32]
  var13339=tf.sin(var13336)
  #[512,32]
  var13340=tf.negative(var13339)
  #[512,32,1]
  var13341=tf.reshape(var13340, [512,32,1])
  #[512,32,2]
  var13342=tf.concat([var13338,var13341], axis=2)
  #[512,32,1,2]
  var13343=tf.reshape(var13342, [512,32,1,2])
  #[512,32,1]
  var13344=tf.reshape(var13339, [512,32,1])
  #[512,32,2]
  var13345=tf.concat([var13344,var13338], axis=2)
  #[512,32,1,2]
  var13346=tf.reshape(var13345, [512,32,1,2])
  #[512,32,2,2]
  var13347=tf.concat([var13343,var13346], axis=2)
  #[512,32,1,2]
  var13348=tf.matmul(var13334, var13347)
  #[512,64]
  var13349=tf.reshape(var13348, [512,64])
  #[512,65]
  var13350=tf.concat([var13332,var13349], axis=1)
  #[512,64]
  var13351=var13350[:,0:64]
  #[512,32,1,2]
  var13352=tf.reshape(var13351, [512,32,1,2])
  #[512,1,32]
  var13353=var13294[:,1:2]
  #[512,32]
  var13354=tf.reshape(var13353, [512,32])
  #[512,32]
  var13355=tf.cos(var13354)
  #[512,32,1]
  var13356=tf.reshape(var13355, [512,32,1])
  #[512,32]
  var13357=tf.sin(var13354)
  #[512,32]
  var13358=tf.negative(var13357)
  #[512,32,1]
  var13359=tf.reshape(var13358, [512,32,1])
  #[512,32,2]
  var13360=tf.concat([var13356,var13359], axis=2)
  #[512,32,1,2]
  var13361=tf.reshape(var13360, [512,32,1,2])
  #[512,32,1]
  var13362=tf.reshape(var13357, [512,32,1])
  #[512,32,2]
  var13363=tf.concat([var13362,var13356], axis=2)
  #[512,32,1,2]
  var13364=tf.reshape(var13363, [512,32,1,2])
  #[512,32,2,2]
  var13365=tf.concat([var13361,var13364], axis=2)
  #[512,32,1,2]
  var13366=tf.matmul(var13352, var13365)
  #[512,64]
  var13367=tf.reshape(var13366, [512,64])
  #[512,1]
  var13368=var13350[:,64:65]
  #[512,1]
  var13369=tf.reshape(var13368, [512,1])
  #[512,65]
  var13370=tf.concat([var13367,var13369], axis=1)
  #[512,1]
  var13371=var13370[:,0:1]
  #[512,1]
  var13372=tf.reshape(var13371, [512,1])
  #[512,64]
  var13373=var13370[:,1:65]
  #[512,32,1,2]
  var13374=tf.reshape(var13373, [512,32,1,2])
  #[512,1,32]
  var13375=var13294[:,0:1]
  #[512,32]
  var13376=tf.reshape(var13375, [512,32])
  #[512,32]
  var13377=tf.cos(var13376)
  #[512,32,1]
  var13378=tf.reshape(var13377, [512,32,1])
  #[512,32]
  var13379=tf.sin(var13376)
  #[512,32]
  var13380=tf.negative(var13379)
  #[512,32,1]
  var13381=tf.reshape(var13380, [512,32,1])
  #[512,32,2]
  var13382=tf.concat([var13378,var13381], axis=2)
  #[512,32,1,2]
  var13383=tf.reshape(var13382, [512,32,1,2])
  #[512,32,1]
  var13384=tf.reshape(var13379, [512,32,1])
  #[512,32,2]
  var13385=tf.concat([var13384,var13378], axis=2)
  #[512,32,1,2]
  var13386=tf.reshape(var13385, [512,32,1,2])
  #[512,32,2,2]
  var13387=tf.concat([var13383,var13386], axis=2)
  #[512,32,1,2]
  var13388=tf.matmul(var13374, var13387)
  #[512,64]
  var13389=tf.reshape(var13388, [512,64])
  #[512,65]
  var13390=tf.concat([var13372,var13389], axis=1)
  #[512,65]
  var13391=tf.multiply(var12364, var13390)
  #[512,65]
  var13392=tf.reshape(var13391, [512,65])
  #[512,12]
  var13393=tf.matmul(var13392, var12494)
  #[512,12]
  var13394=tf.reshape(var13393, [512,12])
  #[512,12]
  var13395=tf.add(var13394, var12498)
  #[512,1,12]
  var13396=tf.reshape(var13395, [512,1,12])
  #[512,65]
  var13397=tf.multiply(var12368, var13390)
  #[512,1]
  var13398=var13397[:,0:1]
  #[512,1]
  var13399=tf.reshape(var13398, [512,1])
  #[512,64]
  var13400=var13397[:,1:65]
  #[512,32,1,2]
  var13401=tf.reshape(var13400, [512,32,1,2])
  #[512,1]
  var13402=var12390[:,9:10]
  #[512]
  var13403=tf.reshape(var13402, [512])
  #[512,160]
  var13404=tf.gather(params=var12389, indices=var13403, batch_dims=0, axis=0)
  #[512,160]
  var13405=tf.multiply(var12388, var13404)
  #[512,5,32]
  var13406=tf.reshape(var13405, [512,5,32])
  #[512,1,32]
  var13407=var13406[:,4:5]
  #[512,32]
  var13408=tf.reshape(var13407, [512,32])
  #[512,32]
  var13409=tf.cos(var13408)
  #[512,32,1]
  var13410=tf.reshape(var13409, [512,32,1])
  #[512,32]
  var13411=tf.sin(var13408)
  #[512,32]
  var13412=tf.negative(var13411)
  #[512,32,1]
  var13413=tf.reshape(var13412, [512,32,1])
  #[512,32,2]
  var13414=tf.concat([var13410,var13413], axis=2)
  #[512,32,1,2]
  var13415=tf.reshape(var13414, [512,32,1,2])
  #[512,32,1]
  var13416=tf.reshape(var13411, [512,32,1])
  #[512,32,2]
  var13417=tf.concat([var13416,var13410], axis=2)
  #[512,32,1,2]
  var13418=tf.reshape(var13417, [512,32,1,2])
  #[512,32,2,2]
  var13419=tf.concat([var13415,var13418], axis=2)
  #[512,32,1,2]
  var13420=tf.matmul(var13401, var13419)
  #[512,64]
  var13421=tf.reshape(var13420, [512,64])
  #[512,65]
  var13422=tf.concat([var13399,var13421], axis=1)
  #[512,64]
  var13423=var13422[:,0:64]
  #[512,32,1,2]
  var13424=tf.reshape(var13423, [512,32,1,2])
  #[512,1,32]
  var13425=var13406[:,3:4]
  #[512,32]
  var13426=tf.reshape(var13425, [512,32])
  #[512,32]
  var13427=tf.cos(var13426)
  #[512,32,1]
  var13428=tf.reshape(var13427, [512,32,1])
  #[512,32]
  var13429=tf.sin(var13426)
  #[512,32]
  var13430=tf.negative(var13429)
  #[512,32,1]
  var13431=tf.reshape(var13430, [512,32,1])
  #[512,32,2]
  var13432=tf.concat([var13428,var13431], axis=2)
  #[512,32,1,2]
  var13433=tf.reshape(var13432, [512,32,1,2])
  #[512,32,1]
  var13434=tf.reshape(var13429, [512,32,1])
  #[512,32,2]
  var13435=tf.concat([var13434,var13428], axis=2)
  #[512,32,1,2]
  var13436=tf.reshape(var13435, [512,32,1,2])
  #[512,32,2,2]
  var13437=tf.concat([var13433,var13436], axis=2)
  #[512,32,1,2]
  var13438=tf.matmul(var13424, var13437)
  #[512,64]
  var13439=tf.reshape(var13438, [512,64])
  #[512,1]
  var13440=var13422[:,64:65]
  #[512,1]
  var13441=tf.reshape(var13440, [512,1])
  #[512,65]
  var13442=tf.concat([var13439,var13441], axis=1)
  #[512,1]
  var13443=var13442[:,0:1]
  #[512,1]
  var13444=tf.reshape(var13443, [512,1])
  #[512,64]
  var13445=var13442[:,1:65]
  #[512,32,1,2]
  var13446=tf.reshape(var13445, [512,32,1,2])
  #[512,1,32]
  var13447=var13406[:,2:3]
  #[512,32]
  var13448=tf.reshape(var13447, [512,32])
  #[512,32]
  var13449=tf.cos(var13448)
  #[512,32,1]
  var13450=tf.reshape(var13449, [512,32,1])
  #[512,32]
  var13451=tf.sin(var13448)
  #[512,32]
  var13452=tf.negative(var13451)
  #[512,32,1]
  var13453=tf.reshape(var13452, [512,32,1])
  #[512,32,2]
  var13454=tf.concat([var13450,var13453], axis=2)
  #[512,32,1,2]
  var13455=tf.reshape(var13454, [512,32,1,2])
  #[512,32,1]
  var13456=tf.reshape(var13451, [512,32,1])
  #[512,32,2]
  var13457=tf.concat([var13456,var13450], axis=2)
  #[512,32,1,2]
  var13458=tf.reshape(var13457, [512,32,1,2])
  #[512,32,2,2]
  var13459=tf.concat([var13455,var13458], axis=2)
  #[512,32,1,2]
  var13460=tf.matmul(var13446, var13459)
  #[512,64]
  var13461=tf.reshape(var13460, [512,64])
  #[512,65]
  var13462=tf.concat([var13444,var13461], axis=1)
  #[512,64]
  var13463=var13462[:,0:64]
  #[512,32,1,2]
  var13464=tf.reshape(var13463, [512,32,1,2])
  #[512,1,32]
  var13465=var13406[:,1:2]
  #[512,32]
  var13466=tf.reshape(var13465, [512,32])
  #[512,32]
  var13467=tf.cos(var13466)
  #[512,32,1]
  var13468=tf.reshape(var13467, [512,32,1])
  #[512,32]
  var13469=tf.sin(var13466)
  #[512,32]
  var13470=tf.negative(var13469)
  #[512,32,1]
  var13471=tf.reshape(var13470, [512,32,1])
  #[512,32,2]
  var13472=tf.concat([var13468,var13471], axis=2)
  #[512,32,1,2]
  var13473=tf.reshape(var13472, [512,32,1,2])
  #[512,32,1]
  var13474=tf.reshape(var13469, [512,32,1])
  #[512,32,2]
  var13475=tf.concat([var13474,var13468], axis=2)
  #[512,32,1,2]
  var13476=tf.reshape(var13475, [512,32,1,2])
  #[512,32,2,2]
  var13477=tf.concat([var13473,var13476], axis=2)
  #[512,32,1,2]
  var13478=tf.matmul(var13464, var13477)
  #[512,64]
  var13479=tf.reshape(var13478, [512,64])
  #[512,1]
  var13480=var13462[:,64:65]
  #[512,1]
  var13481=tf.reshape(var13480, [512,1])
  #[512,65]
  var13482=tf.concat([var13479,var13481], axis=1)
  #[512,1]
  var13483=var13482[:,0:1]
  #[512,1]
  var13484=tf.reshape(var13483, [512,1])
  #[512,64]
  var13485=var13482[:,1:65]
  #[512,32,1,2]
  var13486=tf.reshape(var13485, [512,32,1,2])
  #[512,1,32]
  var13487=var13406[:,0:1]
  #[512,32]
  var13488=tf.reshape(var13487, [512,32])
  #[512,32]
  var13489=tf.cos(var13488)
  #[512,32,1]
  var13490=tf.reshape(var13489, [512,32,1])
  #[512,32]
  var13491=tf.sin(var13488)
  #[512,32]
  var13492=tf.negative(var13491)
  #[512,32,1]
  var13493=tf.reshape(var13492, [512,32,1])
  #[512,32,2]
  var13494=tf.concat([var13490,var13493], axis=2)
  #[512,32,1,2]
  var13495=tf.reshape(var13494, [512,32,1,2])
  #[512,32,1]
  var13496=tf.reshape(var13491, [512,32,1])
  #[512,32,2]
  var13497=tf.concat([var13496,var13490], axis=2)
  #[512,32,1,2]
  var13498=tf.reshape(var13497, [512,32,1,2])
  #[512,32,2,2]
  var13499=tf.concat([var13495,var13498], axis=2)
  #[512,32,1,2]
  var13500=tf.matmul(var13486, var13499)
  #[512,64]
  var13501=tf.reshape(var13500, [512,64])
  #[512,65]
  var13502=tf.concat([var13484,var13501], axis=1)
  #[512,65]
  var13503=tf.multiply(var12364, var13502)
  #[512,65]
  var13504=tf.reshape(var13503, [512,65])
  #[512,12]
  var13505=tf.matmul(var13504, var12494)
  #[512,12]
  var13506=tf.reshape(var13505, [512,12])
  #[512,12]
  var13507=tf.add(var13506, var12498)
  #[512,1,12]
  var13508=tf.reshape(var13507, [512,1,12])
  #[512,65]
  var13509=tf.multiply(var12368, var13502)
  #[512,1]
  var13510=var13509[:,0:1]
  #[512,1]
  var13511=tf.reshape(var13510, [512,1])
  #[512,64]
  var13512=var13509[:,1:65]
  #[512,32,1,2]
  var13513=tf.reshape(var13512, [512,32,1,2])
  #[512,1]
  var13514=var12390[:,10:11]
  #[512]
  var13515=tf.reshape(var13514, [512])
  #[512,160]
  var13516=tf.gather(params=var12389, indices=var13515, batch_dims=0, axis=0)
  #[512,160]
  var13517=tf.multiply(var12388, var13516)
  #[512,5,32]
  var13518=tf.reshape(var13517, [512,5,32])
  #[512,1,32]
  var13519=var13518[:,4:5]
  #[512,32]
  var13520=tf.reshape(var13519, [512,32])
  #[512,32]
  var13521=tf.cos(var13520)
  #[512,32,1]
  var13522=tf.reshape(var13521, [512,32,1])
  #[512,32]
  var13523=tf.sin(var13520)
  #[512,32]
  var13524=tf.negative(var13523)
  #[512,32,1]
  var13525=tf.reshape(var13524, [512,32,1])
  #[512,32,2]
  var13526=tf.concat([var13522,var13525], axis=2)
  #[512,32,1,2]
  var13527=tf.reshape(var13526, [512,32,1,2])
  #[512,32,1]
  var13528=tf.reshape(var13523, [512,32,1])
  #[512,32,2]
  var13529=tf.concat([var13528,var13522], axis=2)
  #[512,32,1,2]
  var13530=tf.reshape(var13529, [512,32,1,2])
  #[512,32,2,2]
  var13531=tf.concat([var13527,var13530], axis=2)
  #[512,32,1,2]
  var13532=tf.matmul(var13513, var13531)
  #[512,64]
  var13533=tf.reshape(var13532, [512,64])
  #[512,65]
  var13534=tf.concat([var13511,var13533], axis=1)
  #[512,64]
  var13535=var13534[:,0:64]
  #[512,32,1,2]
  var13536=tf.reshape(var13535, [512,32,1,2])
  #[512,1,32]
  var13537=var13518[:,3:4]
  #[512,32]
  var13538=tf.reshape(var13537, [512,32])
  #[512,32]
  var13539=tf.cos(var13538)
  #[512,32,1]
  var13540=tf.reshape(var13539, [512,32,1])
  #[512,32]
  var13541=tf.sin(var13538)
  #[512,32]
  var13542=tf.negative(var13541)
  #[512,32,1]
  var13543=tf.reshape(var13542, [512,32,1])
  #[512,32,2]
  var13544=tf.concat([var13540,var13543], axis=2)
  #[512,32,1,2]
  var13545=tf.reshape(var13544, [512,32,1,2])
  #[512,32,1]
  var13546=tf.reshape(var13541, [512,32,1])
  #[512,32,2]
  var13547=tf.concat([var13546,var13540], axis=2)
  #[512,32,1,2]
  var13548=tf.reshape(var13547, [512,32,1,2])
  #[512,32,2,2]
  var13549=tf.concat([var13545,var13548], axis=2)
  #[512,32,1,2]
  var13550=tf.matmul(var13536, var13549)
  #[512,64]
  var13551=tf.reshape(var13550, [512,64])
  #[512,1]
  var13552=var13534[:,64:65]
  #[512,1]
  var13553=tf.reshape(var13552, [512,1])
  #[512,65]
  var13554=tf.concat([var13551,var13553], axis=1)
  #[512,1]
  var13555=var13554[:,0:1]
  #[512,1]
  var13556=tf.reshape(var13555, [512,1])
  #[512,64]
  var13557=var13554[:,1:65]
  #[512,32,1,2]
  var13558=tf.reshape(var13557, [512,32,1,2])
  #[512,1,32]
  var13559=var13518[:,2:3]
  #[512,32]
  var13560=tf.reshape(var13559, [512,32])
  #[512,32]
  var13561=tf.cos(var13560)
  #[512,32,1]
  var13562=tf.reshape(var13561, [512,32,1])
  #[512,32]
  var13563=tf.sin(var13560)
  #[512,32]
  var13564=tf.negative(var13563)
  #[512,32,1]
  var13565=tf.reshape(var13564, [512,32,1])
  #[512,32,2]
  var13566=tf.concat([var13562,var13565], axis=2)
  #[512,32,1,2]
  var13567=tf.reshape(var13566, [512,32,1,2])
  #[512,32,1]
  var13568=tf.reshape(var13563, [512,32,1])
  #[512,32,2]
  var13569=tf.concat([var13568,var13562], axis=2)
  #[512,32,1,2]
  var13570=tf.reshape(var13569, [512,32,1,2])
  #[512,32,2,2]
  var13571=tf.concat([var13567,var13570], axis=2)
  #[512,32,1,2]
  var13572=tf.matmul(var13558, var13571)
  #[512,64]
  var13573=tf.reshape(var13572, [512,64])
  #[512,65]
  var13574=tf.concat([var13556,var13573], axis=1)
  #[512,64]
  var13575=var13574[:,0:64]
  #[512,32,1,2]
  var13576=tf.reshape(var13575, [512,32,1,2])
  #[512,1,32]
  var13577=var13518[:,1:2]
  #[512,32]
  var13578=tf.reshape(var13577, [512,32])
  #[512,32]
  var13579=tf.cos(var13578)
  #[512,32,1]
  var13580=tf.reshape(var13579, [512,32,1])
  #[512,32]
  var13581=tf.sin(var13578)
  #[512,32]
  var13582=tf.negative(var13581)
  #[512,32,1]
  var13583=tf.reshape(var13582, [512,32,1])
  #[512,32,2]
  var13584=tf.concat([var13580,var13583], axis=2)
  #[512,32,1,2]
  var13585=tf.reshape(var13584, [512,32,1,2])
  #[512,32,1]
  var13586=tf.reshape(var13581, [512,32,1])
  #[512,32,2]
  var13587=tf.concat([var13586,var13580], axis=2)
  #[512,32,1,2]
  var13588=tf.reshape(var13587, [512,32,1,2])
  #[512,32,2,2]
  var13589=tf.concat([var13585,var13588], axis=2)
  #[512,32,1,2]
  var13590=tf.matmul(var13576, var13589)
  #[512,64]
  var13591=tf.reshape(var13590, [512,64])
  #[512,1]
  var13592=var13574[:,64:65]
  #[512,1]
  var13593=tf.reshape(var13592, [512,1])
  #[512,65]
  var13594=tf.concat([var13591,var13593], axis=1)
  #[512,1]
  var13595=var13594[:,0:1]
  #[512,1]
  var13596=tf.reshape(var13595, [512,1])
  #[512,64]
  var13597=var13594[:,1:65]
  #[512,32,1,2]
  var13598=tf.reshape(var13597, [512,32,1,2])
  #[512,1,32]
  var13599=var13518[:,0:1]
  #[512,32]
  var13600=tf.reshape(var13599, [512,32])
  #[512,32]
  var13601=tf.cos(var13600)
  #[512,32,1]
  var13602=tf.reshape(var13601, [512,32,1])
  #[512,32]
  var13603=tf.sin(var13600)
  #[512,32]
  var13604=tf.negative(var13603)
  #[512,32,1]
  var13605=tf.reshape(var13604, [512,32,1])
  #[512,32,2]
  var13606=tf.concat([var13602,var13605], axis=2)
  #[512,32,1,2]
  var13607=tf.reshape(var13606, [512,32,1,2])
  #[512,32,1]
  var13608=tf.reshape(var13603, [512,32,1])
  #[512,32,2]
  var13609=tf.concat([var13608,var13602], axis=2)
  #[512,32,1,2]
  var13610=tf.reshape(var13609, [512,32,1,2])
  #[512,32,2,2]
  var13611=tf.concat([var13607,var13610], axis=2)
  #[512,32,1,2]
  var13612=tf.matmul(var13598, var13611)
  #[512,64]
  var13613=tf.reshape(var13612, [512,64])
  #[512,65]
  var13614=tf.concat([var13596,var13613], axis=1)
  #[512,65]
  var13615=tf.multiply(var12364, var13614)
  #[512,65]
  var13616=tf.reshape(var13615, [512,65])
  #[512,12]
  var13617=tf.matmul(var13616, var12494)
  #[512,12]
  var13618=tf.reshape(var13617, [512,12])
  #[512,12]
  var13619=tf.add(var13618, var12498)
  #[512,1,12]
  var13620=tf.reshape(var13619, [512,1,12])
  #[512,65]
  var13621=tf.multiply(var12368, var13614)
  #[512,1]
  var13622=var13621[:,0:1]
  #[512,1]
  var13623=tf.reshape(var13622, [512,1])
  #[512,64]
  var13624=var13621[:,1:65]
  #[512,32,1,2]
  var13625=tf.reshape(var13624, [512,32,1,2])
  #[512,1]
  var13626=var12390[:,11:12]
  #[512]
  var13627=tf.reshape(var13626, [512])
  #[512,160]
  var13628=tf.gather(params=var12389, indices=var13627, batch_dims=0, axis=0)
  #[512,160]
  var13629=tf.multiply(var12388, var13628)
  #[512,5,32]
  var13630=tf.reshape(var13629, [512,5,32])
  #[512,1,32]
  var13631=var13630[:,4:5]
  #[512,32]
  var13632=tf.reshape(var13631, [512,32])
  #[512,32]
  var13633=tf.cos(var13632)
  #[512,32,1]
  var13634=tf.reshape(var13633, [512,32,1])
  #[512,32]
  var13635=tf.sin(var13632)
  #[512,32]
  var13636=tf.negative(var13635)
  #[512,32,1]
  var13637=tf.reshape(var13636, [512,32,1])
  #[512,32,2]
  var13638=tf.concat([var13634,var13637], axis=2)
  #[512,32,1,2]
  var13639=tf.reshape(var13638, [512,32,1,2])
  #[512,32,1]
  var13640=tf.reshape(var13635, [512,32,1])
  #[512,32,2]
  var13641=tf.concat([var13640,var13634], axis=2)
  #[512,32,1,2]
  var13642=tf.reshape(var13641, [512,32,1,2])
  #[512,32,2,2]
  var13643=tf.concat([var13639,var13642], axis=2)
  #[512,32,1,2]
  var13644=tf.matmul(var13625, var13643)
  #[512,64]
  var13645=tf.reshape(var13644, [512,64])
  #[512,65]
  var13646=tf.concat([var13623,var13645], axis=1)
  #[512,64]
  var13647=var13646[:,0:64]
  #[512,32,1,2]
  var13648=tf.reshape(var13647, [512,32,1,2])
  #[512,1,32]
  var13649=var13630[:,3:4]
  #[512,32]
  var13650=tf.reshape(var13649, [512,32])
  #[512,32]
  var13651=tf.cos(var13650)
  #[512,32,1]
  var13652=tf.reshape(var13651, [512,32,1])
  #[512,32]
  var13653=tf.sin(var13650)
  #[512,32]
  var13654=tf.negative(var13653)
  #[512,32,1]
  var13655=tf.reshape(var13654, [512,32,1])
  #[512,32,2]
  var13656=tf.concat([var13652,var13655], axis=2)
  #[512,32,1,2]
  var13657=tf.reshape(var13656, [512,32,1,2])
  #[512,32,1]
  var13658=tf.reshape(var13653, [512,32,1])
  #[512,32,2]
  var13659=tf.concat([var13658,var13652], axis=2)
  #[512,32,1,2]
  var13660=tf.reshape(var13659, [512,32,1,2])
  #[512,32,2,2]
  var13661=tf.concat([var13657,var13660], axis=2)
  #[512,32,1,2]
  var13662=tf.matmul(var13648, var13661)
  #[512,64]
  var13663=tf.reshape(var13662, [512,64])
  #[512,1]
  var13664=var13646[:,64:65]
  #[512,1]
  var13665=tf.reshape(var13664, [512,1])
  #[512,65]
  var13666=tf.concat([var13663,var13665], axis=1)
  #[512,1]
  var13667=var13666[:,0:1]
  #[512,1]
  var13668=tf.reshape(var13667, [512,1])
  #[512,64]
  var13669=var13666[:,1:65]
  #[512,32,1,2]
  var13670=tf.reshape(var13669, [512,32,1,2])
  #[512,1,32]
  var13671=var13630[:,2:3]
  #[512,32]
  var13672=tf.reshape(var13671, [512,32])
  #[512,32]
  var13673=tf.cos(var13672)
  #[512,32,1]
  var13674=tf.reshape(var13673, [512,32,1])
  #[512,32]
  var13675=tf.sin(var13672)
  #[512,32]
  var13676=tf.negative(var13675)
  #[512,32,1]
  var13677=tf.reshape(var13676, [512,32,1])
  #[512,32,2]
  var13678=tf.concat([var13674,var13677], axis=2)
  #[512,32,1,2]
  var13679=tf.reshape(var13678, [512,32,1,2])
  #[512,32,1]
  var13680=tf.reshape(var13675, [512,32,1])
  #[512,32,2]
  var13681=tf.concat([var13680,var13674], axis=2)
  #[512,32,1,2]
  var13682=tf.reshape(var13681, [512,32,1,2])
  #[512,32,2,2]
  var13683=tf.concat([var13679,var13682], axis=2)
  #[512,32,1,2]
  var13684=tf.matmul(var13670, var13683)
  #[512,64]
  var13685=tf.reshape(var13684, [512,64])
  #[512,65]
  var13686=tf.concat([var13668,var13685], axis=1)
  #[512,64]
  var13687=var13686[:,0:64]
  #[512,32,1,2]
  var13688=tf.reshape(var13687, [512,32,1,2])
  #[512,1,32]
  var13689=var13630[:,1:2]
  #[512,32]
  var13690=tf.reshape(var13689, [512,32])
  #[512,32]
  var13691=tf.cos(var13690)
  #[512,32,1]
  var13692=tf.reshape(var13691, [512,32,1])
  #[512,32]
  var13693=tf.sin(var13690)
  #[512,32]
  var13694=tf.negative(var13693)
  #[512,32,1]
  var13695=tf.reshape(var13694, [512,32,1])
  #[512,32,2]
  var13696=tf.concat([var13692,var13695], axis=2)
  #[512,32,1,2]
  var13697=tf.reshape(var13696, [512,32,1,2])
  #[512,32,1]
  var13698=tf.reshape(var13693, [512,32,1])
  #[512,32,2]
  var13699=tf.concat([var13698,var13692], axis=2)
  #[512,32,1,2]
  var13700=tf.reshape(var13699, [512,32,1,2])
  #[512,32,2,2]
  var13701=tf.concat([var13697,var13700], axis=2)
  #[512,32,1,2]
  var13702=tf.matmul(var13688, var13701)
  #[512,64]
  var13703=tf.reshape(var13702, [512,64])
  #[512,1]
  var13704=var13686[:,64:65]
  #[512,1]
  var13705=tf.reshape(var13704, [512,1])
  #[512,65]
  var13706=tf.concat([var13703,var13705], axis=1)
  #[512,1]
  var13707=var13706[:,0:1]
  #[512,1]
  var13708=tf.reshape(var13707, [512,1])
  #[512,64]
  var13709=var13706[:,1:65]
  #[512,32,1,2]
  var13710=tf.reshape(var13709, [512,32,1,2])
  #[512,1,32]
  var13711=var13630[:,0:1]
  #[512,32]
  var13712=tf.reshape(var13711, [512,32])
  #[512,32]
  var13713=tf.cos(var13712)
  #[512,32,1]
  var13714=tf.reshape(var13713, [512,32,1])
  #[512,32]
  var13715=tf.sin(var13712)
  #[512,32]
  var13716=tf.negative(var13715)
  #[512,32,1]
  var13717=tf.reshape(var13716, [512,32,1])
  #[512,32,2]
  var13718=tf.concat([var13714,var13717], axis=2)
  #[512,32,1,2]
  var13719=tf.reshape(var13718, [512,32,1,2])
  #[512,32,1]
  var13720=tf.reshape(var13715, [512,32,1])
  #[512,32,2]
  var13721=tf.concat([var13720,var13714], axis=2)
  #[512,32,1,2]
  var13722=tf.reshape(var13721, [512,32,1,2])
  #[512,32,2,2]
  var13723=tf.concat([var13719,var13722], axis=2)
  #[512,32,1,2]
  var13724=tf.matmul(var13710, var13723)
  #[512,64]
  var13725=tf.reshape(var13724, [512,64])
  #[512,65]
  var13726=tf.concat([var13708,var13725], axis=1)
  #[512,65]
  var13727=tf.multiply(var12364, var13726)
  #[512,65]
  var13728=tf.reshape(var13727, [512,65])
  #[512,12]
  var13729=tf.matmul(var13728, var12494)
  #[512,12]
  var13730=tf.reshape(var13729, [512,12])
  #[512,12]
  var13731=tf.add(var13730, var12498)
  #[512,1,12]
  var13732=tf.reshape(var13731, [512,1,12])
  #[512,65]
  var13733=tf.multiply(var12368, var13726)
  #[512,1]
  var13734=var13733[:,0:1]
  #[512,1]
  var13735=tf.reshape(var13734, [512,1])
  #[512,64]
  var13736=var13733[:,1:65]
  #[512,32,1,2]
  var13737=tf.reshape(var13736, [512,32,1,2])
  #[512,1]
  var13738=var12390[:,12:13]
  #[512]
  var13739=tf.reshape(var13738, [512])
  #[512,160]
  var13740=tf.gather(params=var12389, indices=var13739, batch_dims=0, axis=0)
  #[512,160]
  var13741=tf.multiply(var12388, var13740)
  #[512,5,32]
  var13742=tf.reshape(var13741, [512,5,32])
  #[512,1,32]
  var13743=var13742[:,4:5]
  #[512,32]
  var13744=tf.reshape(var13743, [512,32])
  #[512,32]
  var13745=tf.cos(var13744)
  #[512,32,1]
  var13746=tf.reshape(var13745, [512,32,1])
  #[512,32]
  var13747=tf.sin(var13744)
  #[512,32]
  var13748=tf.negative(var13747)
  #[512,32,1]
  var13749=tf.reshape(var13748, [512,32,1])
  #[512,32,2]
  var13750=tf.concat([var13746,var13749], axis=2)
  #[512,32,1,2]
  var13751=tf.reshape(var13750, [512,32,1,2])
  #[512,32,1]
  var13752=tf.reshape(var13747, [512,32,1])
  #[512,32,2]
  var13753=tf.concat([var13752,var13746], axis=2)
  #[512,32,1,2]
  var13754=tf.reshape(var13753, [512,32,1,2])
  #[512,32,2,2]
  var13755=tf.concat([var13751,var13754], axis=2)
  #[512,32,1,2]
  var13756=tf.matmul(var13737, var13755)
  #[512,64]
  var13757=tf.reshape(var13756, [512,64])
  #[512,65]
  var13758=tf.concat([var13735,var13757], axis=1)
  #[512,64]
  var13759=var13758[:,0:64]
  #[512,32,1,2]
  var13760=tf.reshape(var13759, [512,32,1,2])
  #[512,1,32]
  var13761=var13742[:,3:4]
  #[512,32]
  var13762=tf.reshape(var13761, [512,32])
  #[512,32]
  var13763=tf.cos(var13762)
  #[512,32,1]
  var13764=tf.reshape(var13763, [512,32,1])
  #[512,32]
  var13765=tf.sin(var13762)
  #[512,32]
  var13766=tf.negative(var13765)
  #[512,32,1]
  var13767=tf.reshape(var13766, [512,32,1])
  #[512,32,2]
  var13768=tf.concat([var13764,var13767], axis=2)
  #[512,32,1,2]
  var13769=tf.reshape(var13768, [512,32,1,2])
  #[512,32,1]
  var13770=tf.reshape(var13765, [512,32,1])
  #[512,32,2]
  var13771=tf.concat([var13770,var13764], axis=2)
  #[512,32,1,2]
  var13772=tf.reshape(var13771, [512,32,1,2])
  #[512,32,2,2]
  var13773=tf.concat([var13769,var13772], axis=2)
  #[512,32,1,2]
  var13774=tf.matmul(var13760, var13773)
  #[512,64]
  var13775=tf.reshape(var13774, [512,64])
  #[512,1]
  var13776=var13758[:,64:65]
  #[512,1]
  var13777=tf.reshape(var13776, [512,1])
  #[512,65]
  var13778=tf.concat([var13775,var13777], axis=1)
  #[512,1]
  var13779=var13778[:,0:1]
  #[512,1]
  var13780=tf.reshape(var13779, [512,1])
  #[512,64]
  var13781=var13778[:,1:65]
  #[512,32,1,2]
  var13782=tf.reshape(var13781, [512,32,1,2])
  #[512,1,32]
  var13783=var13742[:,2:3]
  #[512,32]
  var13784=tf.reshape(var13783, [512,32])
  #[512,32]
  var13785=tf.cos(var13784)
  #[512,32,1]
  var13786=tf.reshape(var13785, [512,32,1])
  #[512,32]
  var13787=tf.sin(var13784)
  #[512,32]
  var13788=tf.negative(var13787)
  #[512,32,1]
  var13789=tf.reshape(var13788, [512,32,1])
  #[512,32,2]
  var13790=tf.concat([var13786,var13789], axis=2)
  #[512,32,1,2]
  var13791=tf.reshape(var13790, [512,32,1,2])
  #[512,32,1]
  var13792=tf.reshape(var13787, [512,32,1])
  #[512,32,2]
  var13793=tf.concat([var13792,var13786], axis=2)
  #[512,32,1,2]
  var13794=tf.reshape(var13793, [512,32,1,2])
  #[512,32,2,2]
  var13795=tf.concat([var13791,var13794], axis=2)
  #[512,32,1,2]
  var13796=tf.matmul(var13782, var13795)
  #[512,64]
  var13797=tf.reshape(var13796, [512,64])
  #[512,65]
  var13798=tf.concat([var13780,var13797], axis=1)
  #[512,64]
  var13799=var13798[:,0:64]
  #[512,32,1,2]
  var13800=tf.reshape(var13799, [512,32,1,2])
  #[512,1,32]
  var13801=var13742[:,1:2]
  #[512,32]
  var13802=tf.reshape(var13801, [512,32])
  #[512,32]
  var13803=tf.cos(var13802)
  #[512,32,1]
  var13804=tf.reshape(var13803, [512,32,1])
  #[512,32]
  var13805=tf.sin(var13802)
  #[512,32]
  var13806=tf.negative(var13805)
  #[512,32,1]
  var13807=tf.reshape(var13806, [512,32,1])
  #[512,32,2]
  var13808=tf.concat([var13804,var13807], axis=2)
  #[512,32,1,2]
  var13809=tf.reshape(var13808, [512,32,1,2])
  #[512,32,1]
  var13810=tf.reshape(var13805, [512,32,1])
  #[512,32,2]
  var13811=tf.concat([var13810,var13804], axis=2)
  #[512,32,1,2]
  var13812=tf.reshape(var13811, [512,32,1,2])
  #[512,32,2,2]
  var13813=tf.concat([var13809,var13812], axis=2)
  #[512,32,1,2]
  var13814=tf.matmul(var13800, var13813)
  #[512,64]
  var13815=tf.reshape(var13814, [512,64])
  #[512,1]
  var13816=var13798[:,64:65]
  #[512,1]
  var13817=tf.reshape(var13816, [512,1])
  #[512,65]
  var13818=tf.concat([var13815,var13817], axis=1)
  #[512,1]
  var13819=var13818[:,0:1]
  #[512,1]
  var13820=tf.reshape(var13819, [512,1])
  #[512,64]
  var13821=var13818[:,1:65]
  #[512,32,1,2]
  var13822=tf.reshape(var13821, [512,32,1,2])
  #[512,1,32]
  var13823=var13742[:,0:1]
  #[512,32]
  var13824=tf.reshape(var13823, [512,32])
  #[512,32]
  var13825=tf.cos(var13824)
  #[512,32,1]
  var13826=tf.reshape(var13825, [512,32,1])
  #[512,32]
  var13827=tf.sin(var13824)
  #[512,32]
  var13828=tf.negative(var13827)
  #[512,32,1]
  var13829=tf.reshape(var13828, [512,32,1])
  #[512,32,2]
  var13830=tf.concat([var13826,var13829], axis=2)
  #[512,32,1,2]
  var13831=tf.reshape(var13830, [512,32,1,2])
  #[512,32,1]
  var13832=tf.reshape(var13827, [512,32,1])
  #[512,32,2]
  var13833=tf.concat([var13832,var13826], axis=2)
  #[512,32,1,2]
  var13834=tf.reshape(var13833, [512,32,1,2])
  #[512,32,2,2]
  var13835=tf.concat([var13831,var13834], axis=2)
  #[512,32,1,2]
  var13836=tf.matmul(var13822, var13835)
  #[512,64]
  var13837=tf.reshape(var13836, [512,64])
  #[512,65]
  var13838=tf.concat([var13820,var13837], axis=1)
  #[512,65]
  var13839=tf.multiply(var12364, var13838)
  #[512,65]
  var13840=tf.reshape(var13839, [512,65])
  #[512,12]
  var13841=tf.matmul(var13840, var12494)
  #[512,12]
  var13842=tf.reshape(var13841, [512,12])
  #[512,12]
  var13843=tf.add(var13842, var12498)
  #[512,1,12]
  var13844=tf.reshape(var13843, [512,1,12])
  #[512,65]
  var13845=tf.multiply(var12368, var13838)
  #[512,1]
  var13846=var13845[:,0:1]
  #[512,1]
  var13847=tf.reshape(var13846, [512,1])
  #[512,64]
  var13848=var13845[:,1:65]
  #[512,32,1,2]
  var13849=tf.reshape(var13848, [512,32,1,2])
  #[512,1]
  var13850=var12390[:,13:14]
  #[512]
  var13851=tf.reshape(var13850, [512])
  #[512,160]
  var13852=tf.gather(params=var12389, indices=var13851, batch_dims=0, axis=0)
  #[512,160]
  var13853=tf.multiply(var12388, var13852)
  #[512,5,32]
  var13854=tf.reshape(var13853, [512,5,32])
  #[512,1,32]
  var13855=var13854[:,4:5]
  #[512,32]
  var13856=tf.reshape(var13855, [512,32])
  #[512,32]
  var13857=tf.cos(var13856)
  #[512,32,1]
  var13858=tf.reshape(var13857, [512,32,1])
  #[512,32]
  var13859=tf.sin(var13856)
  #[512,32]
  var13860=tf.negative(var13859)
  #[512,32,1]
  var13861=tf.reshape(var13860, [512,32,1])
  #[512,32,2]
  var13862=tf.concat([var13858,var13861], axis=2)
  #[512,32,1,2]
  var13863=tf.reshape(var13862, [512,32,1,2])
  #[512,32,1]
  var13864=tf.reshape(var13859, [512,32,1])
  #[512,32,2]
  var13865=tf.concat([var13864,var13858], axis=2)
  #[512,32,1,2]
  var13866=tf.reshape(var13865, [512,32,1,2])
  #[512,32,2,2]
  var13867=tf.concat([var13863,var13866], axis=2)
  #[512,32,1,2]
  var13868=tf.matmul(var13849, var13867)
  #[512,64]
  var13869=tf.reshape(var13868, [512,64])
  #[512,65]
  var13870=tf.concat([var13847,var13869], axis=1)
  #[512,64]
  var13871=var13870[:,0:64]
  #[512,32,1,2]
  var13872=tf.reshape(var13871, [512,32,1,2])
  #[512,1,32]
  var13873=var13854[:,3:4]
  #[512,32]
  var13874=tf.reshape(var13873, [512,32])
  #[512,32]
  var13875=tf.cos(var13874)
  #[512,32,1]
  var13876=tf.reshape(var13875, [512,32,1])
  #[512,32]
  var13877=tf.sin(var13874)
  #[512,32]
  var13878=tf.negative(var13877)
  #[512,32,1]
  var13879=tf.reshape(var13878, [512,32,1])
  #[512,32,2]
  var13880=tf.concat([var13876,var13879], axis=2)
  #[512,32,1,2]
  var13881=tf.reshape(var13880, [512,32,1,2])
  #[512,32,1]
  var13882=tf.reshape(var13877, [512,32,1])
  #[512,32,2]
  var13883=tf.concat([var13882,var13876], axis=2)
  #[512,32,1,2]
  var13884=tf.reshape(var13883, [512,32,1,2])
  #[512,32,2,2]
  var13885=tf.concat([var13881,var13884], axis=2)
  #[512,32,1,2]
  var13886=tf.matmul(var13872, var13885)
  #[512,64]
  var13887=tf.reshape(var13886, [512,64])
  #[512,1]
  var13888=var13870[:,64:65]
  #[512,1]
  var13889=tf.reshape(var13888, [512,1])
  #[512,65]
  var13890=tf.concat([var13887,var13889], axis=1)
  #[512,1]
  var13891=var13890[:,0:1]
  #[512,1]
  var13892=tf.reshape(var13891, [512,1])
  #[512,64]
  var13893=var13890[:,1:65]
  #[512,32,1,2]
  var13894=tf.reshape(var13893, [512,32,1,2])
  #[512,1,32]
  var13895=var13854[:,2:3]
  #[512,32]
  var13896=tf.reshape(var13895, [512,32])
  #[512,32]
  var13897=tf.cos(var13896)
  #[512,32,1]
  var13898=tf.reshape(var13897, [512,32,1])
  #[512,32]
  var13899=tf.sin(var13896)
  #[512,32]
  var13900=tf.negative(var13899)
  #[512,32,1]
  var13901=tf.reshape(var13900, [512,32,1])
  #[512,32,2]
  var13902=tf.concat([var13898,var13901], axis=2)
  #[512,32,1,2]
  var13903=tf.reshape(var13902, [512,32,1,2])
  #[512,32,1]
  var13904=tf.reshape(var13899, [512,32,1])
  #[512,32,2]
  var13905=tf.concat([var13904,var13898], axis=2)
  #[512,32,1,2]
  var13906=tf.reshape(var13905, [512,32,1,2])
  #[512,32,2,2]
  var13907=tf.concat([var13903,var13906], axis=2)
  #[512,32,1,2]
  var13908=tf.matmul(var13894, var13907)
  #[512,64]
  var13909=tf.reshape(var13908, [512,64])
  #[512,65]
  var13910=tf.concat([var13892,var13909], axis=1)
  #[512,64]
  var13911=var13910[:,0:64]
  #[512,32,1,2]
  var13912=tf.reshape(var13911, [512,32,1,2])
  #[512,1,32]
  var13913=var13854[:,1:2]
  #[512,32]
  var13914=tf.reshape(var13913, [512,32])
  #[512,32]
  var13915=tf.cos(var13914)
  #[512,32,1]
  var13916=tf.reshape(var13915, [512,32,1])
  #[512,32]
  var13917=tf.sin(var13914)
  #[512,32]
  var13918=tf.negative(var13917)
  #[512,32,1]
  var13919=tf.reshape(var13918, [512,32,1])
  #[512,32,2]
  var13920=tf.concat([var13916,var13919], axis=2)
  #[512,32,1,2]
  var13921=tf.reshape(var13920, [512,32,1,2])
  #[512,32,1]
  var13922=tf.reshape(var13917, [512,32,1])
  #[512,32,2]
  var13923=tf.concat([var13922,var13916], axis=2)
  #[512,32,1,2]
  var13924=tf.reshape(var13923, [512,32,1,2])
  #[512,32,2,2]
  var13925=tf.concat([var13921,var13924], axis=2)
  #[512,32,1,2]
  var13926=tf.matmul(var13912, var13925)
  #[512,64]
  var13927=tf.reshape(var13926, [512,64])
  #[512,1]
  var13928=var13910[:,64:65]
  #[512,1]
  var13929=tf.reshape(var13928, [512,1])
  #[512,65]
  var13930=tf.concat([var13927,var13929], axis=1)
  #[512,1]
  var13931=var13930[:,0:1]
  #[512,1]
  var13932=tf.reshape(var13931, [512,1])
  #[512,64]
  var13933=var13930[:,1:65]
  #[512,32,1,2]
  var13934=tf.reshape(var13933, [512,32,1,2])
  #[512,1,32]
  var13935=var13854[:,0:1]
  #[512,32]
  var13936=tf.reshape(var13935, [512,32])
  #[512,32]
  var13937=tf.cos(var13936)
  #[512,32,1]
  var13938=tf.reshape(var13937, [512,32,1])
  #[512,32]
  var13939=tf.sin(var13936)
  #[512,32]
  var13940=tf.negative(var13939)
  #[512,32,1]
  var13941=tf.reshape(var13940, [512,32,1])
  #[512,32,2]
  var13942=tf.concat([var13938,var13941], axis=2)
  #[512,32,1,2]
  var13943=tf.reshape(var13942, [512,32,1,2])
  #[512,32,1]
  var13944=tf.reshape(var13939, [512,32,1])
  #[512,32,2]
  var13945=tf.concat([var13944,var13938], axis=2)
  #[512,32,1,2]
  var13946=tf.reshape(var13945, [512,32,1,2])
  #[512,32,2,2]
  var13947=tf.concat([var13943,var13946], axis=2)
  #[512,32,1,2]
  var13948=tf.matmul(var13934, var13947)
  #[512,64]
  var13949=tf.reshape(var13948, [512,64])
  #[512,65]
  var13950=tf.concat([var13932,var13949], axis=1)
  #[512,65]
  var13951=tf.multiply(var12364, var13950)
  #[512,65]
  var13952=tf.reshape(var13951, [512,65])
  #[512,12]
  var13953=tf.matmul(var13952, var12494)
  #[512,12]
  var13954=tf.reshape(var13953, [512,12])
  #[512,12]
  var13955=tf.add(var13954, var12498)
  #[512,1,12]
  var13956=tf.reshape(var13955, [512,1,12])
  #[512,65]
  var13957=tf.multiply(var12368, var13950)
  #[512,1]
  var13958=var13957[:,0:1]
  #[512,1]
  var13959=tf.reshape(var13958, [512,1])
  #[512,64]
  var13960=var13957[:,1:65]
  #[512,32,1,2]
  var13961=tf.reshape(var13960, [512,32,1,2])
  #[512,1]
  var13962=var12390[:,14:15]
  #[512]
  var13963=tf.reshape(var13962, [512])
  #[512,160]
  var13964=tf.gather(params=var12389, indices=var13963, batch_dims=0, axis=0)
  #[512,160]
  var13965=tf.multiply(var12388, var13964)
  #[512,5,32]
  var13966=tf.reshape(var13965, [512,5,32])
  #[512,1,32]
  var13967=var13966[:,4:5]
  #[512,32]
  var13968=tf.reshape(var13967, [512,32])
  #[512,32]
  var13969=tf.cos(var13968)
  #[512,32,1]
  var13970=tf.reshape(var13969, [512,32,1])
  #[512,32]
  var13971=tf.sin(var13968)
  #[512,32]
  var13972=tf.negative(var13971)
  #[512,32,1]
  var13973=tf.reshape(var13972, [512,32,1])
  #[512,32,2]
  var13974=tf.concat([var13970,var13973], axis=2)
  #[512,32,1,2]
  var13975=tf.reshape(var13974, [512,32,1,2])
  #[512,32,1]
  var13976=tf.reshape(var13971, [512,32,1])
  #[512,32,2]
  var13977=tf.concat([var13976,var13970], axis=2)
  #[512,32,1,2]
  var13978=tf.reshape(var13977, [512,32,1,2])
  #[512,32,2,2]
  var13979=tf.concat([var13975,var13978], axis=2)
  #[512,32,1,2]
  var13980=tf.matmul(var13961, var13979)
  #[512,64]
  var13981=tf.reshape(var13980, [512,64])
  #[512,65]
  var13982=tf.concat([var13959,var13981], axis=1)
  #[512,64]
  var13983=var13982[:,0:64]
  #[512,32,1,2]
  var13984=tf.reshape(var13983, [512,32,1,2])
  #[512,1,32]
  var13985=var13966[:,3:4]
  #[512,32]
  var13986=tf.reshape(var13985, [512,32])
  #[512,32]
  var13987=tf.cos(var13986)
  #[512,32,1]
  var13988=tf.reshape(var13987, [512,32,1])
  #[512,32]
  var13989=tf.sin(var13986)
  #[512,32]
  var13990=tf.negative(var13989)
  #[512,32,1]
  var13991=tf.reshape(var13990, [512,32,1])
  #[512,32,2]
  var13992=tf.concat([var13988,var13991], axis=2)
  #[512,32,1,2]
  var13993=tf.reshape(var13992, [512,32,1,2])
  #[512,32,1]
  var13994=tf.reshape(var13989, [512,32,1])
  #[512,32,2]
  var13995=tf.concat([var13994,var13988], axis=2)
  #[512,32,1,2]
  var13996=tf.reshape(var13995, [512,32,1,2])
  #[512,32,2,2]
  var13997=tf.concat([var13993,var13996], axis=2)
  #[512,32,1,2]
  var13998=tf.matmul(var13984, var13997)
  #[512,64]
  var13999=tf.reshape(var13998, [512,64])
  #[512,1]
  var14000=var13982[:,64:65]
  #[512,1]
  var14001=tf.reshape(var14000, [512,1])
  #[512,65]
  var14002=tf.concat([var13999,var14001], axis=1)
  #[512,1]
  var14003=var14002[:,0:1]
  #[512,1]
  var14004=tf.reshape(var14003, [512,1])
  #[512,64]
  var14005=var14002[:,1:65]
  #[512,32,1,2]
  var14006=tf.reshape(var14005, [512,32,1,2])
  #[512,1,32]
  var14007=var13966[:,2:3]
  #[512,32]
  var14008=tf.reshape(var14007, [512,32])
  #[512,32]
  var14009=tf.cos(var14008)
  #[512,32,1]
  var14010=tf.reshape(var14009, [512,32,1])
  #[512,32]
  var14011=tf.sin(var14008)
  #[512,32]
  var14012=tf.negative(var14011)
  #[512,32,1]
  var14013=tf.reshape(var14012, [512,32,1])
  #[512,32,2]
  var14014=tf.concat([var14010,var14013], axis=2)
  #[512,32,1,2]
  var14015=tf.reshape(var14014, [512,32,1,2])
  #[512,32,1]
  var14016=tf.reshape(var14011, [512,32,1])
  #[512,32,2]
  var14017=tf.concat([var14016,var14010], axis=2)
  #[512,32,1,2]
  var14018=tf.reshape(var14017, [512,32,1,2])
  #[512,32,2,2]
  var14019=tf.concat([var14015,var14018], axis=2)
  #[512,32,1,2]
  var14020=tf.matmul(var14006, var14019)
  #[512,64]
  var14021=tf.reshape(var14020, [512,64])
  #[512,65]
  var14022=tf.concat([var14004,var14021], axis=1)
  #[512,64]
  var14023=var14022[:,0:64]
  #[512,32,1,2]
  var14024=tf.reshape(var14023, [512,32,1,2])
  #[512,1,32]
  var14025=var13966[:,1:2]
  #[512,32]
  var14026=tf.reshape(var14025, [512,32])
  #[512,32]
  var14027=tf.cos(var14026)
  #[512,32,1]
  var14028=tf.reshape(var14027, [512,32,1])
  #[512,32]
  var14029=tf.sin(var14026)
  #[512,32]
  var14030=tf.negative(var14029)
  #[512,32,1]
  var14031=tf.reshape(var14030, [512,32,1])
  #[512,32,2]
  var14032=tf.concat([var14028,var14031], axis=2)
  #[512,32,1,2]
  var14033=tf.reshape(var14032, [512,32,1,2])
  #[512,32,1]
  var14034=tf.reshape(var14029, [512,32,1])
  #[512,32,2]
  var14035=tf.concat([var14034,var14028], axis=2)
  #[512,32,1,2]
  var14036=tf.reshape(var14035, [512,32,1,2])
  #[512,32,2,2]
  var14037=tf.concat([var14033,var14036], axis=2)
  #[512,32,1,2]
  var14038=tf.matmul(var14024, var14037)
  #[512,64]
  var14039=tf.reshape(var14038, [512,64])
  #[512,1]
  var14040=var14022[:,64:65]
  #[512,1]
  var14041=tf.reshape(var14040, [512,1])
  #[512,65]
  var14042=tf.concat([var14039,var14041], axis=1)
  #[512,1]
  var14043=var14042[:,0:1]
  #[512,1]
  var14044=tf.reshape(var14043, [512,1])
  #[512,64]
  var14045=var14042[:,1:65]
  #[512,32,1,2]
  var14046=tf.reshape(var14045, [512,32,1,2])
  #[512,1,32]
  var14047=var13966[:,0:1]
  #[512,32]
  var14048=tf.reshape(var14047, [512,32])
  #[512,32]
  var14049=tf.cos(var14048)
  #[512,32,1]
  var14050=tf.reshape(var14049, [512,32,1])
  #[512,32]
  var14051=tf.sin(var14048)
  #[512,32]
  var14052=tf.negative(var14051)
  #[512,32,1]
  var14053=tf.reshape(var14052, [512,32,1])
  #[512,32,2]
  var14054=tf.concat([var14050,var14053], axis=2)
  #[512,32,1,2]
  var14055=tf.reshape(var14054, [512,32,1,2])
  #[512,32,1]
  var14056=tf.reshape(var14051, [512,32,1])
  #[512,32,2]
  var14057=tf.concat([var14056,var14050], axis=2)
  #[512,32,1,2]
  var14058=tf.reshape(var14057, [512,32,1,2])
  #[512,32,2,2]
  var14059=tf.concat([var14055,var14058], axis=2)
  #[512,32,1,2]
  var14060=tf.matmul(var14046, var14059)
  #[512,64]
  var14061=tf.reshape(var14060, [512,64])
  #[512,65]
  var14062=tf.concat([var14044,var14061], axis=1)
  #[512,65]
  var14063=tf.multiply(var12364, var14062)
  #[512,65]
  var14064=tf.reshape(var14063, [512,65])
  #[512,12]
  var14065=tf.matmul(var14064, var12494)
  #[512,12]
  var14066=tf.reshape(var14065, [512,12])
  #[512,12]
  var14067=tf.add(var14066, var12498)
  #[512,1,12]
  var14068=tf.reshape(var14067, [512,1,12])
  #[512,65]
  var14069=tf.multiply(var12368, var14062)
  #[512,1]
  var14070=var14069[:,0:1]
  #[512,1]
  var14071=tf.reshape(var14070, [512,1])
  #[512,64]
  var14072=var14069[:,1:65]
  #[512,32,1,2]
  var14073=tf.reshape(var14072, [512,32,1,2])
  #[512,1]
  var14074=var12390[:,15:16]
  #[512]
  var14075=tf.reshape(var14074, [512])
  #[512,160]
  var14076=tf.gather(params=var12389, indices=var14075, batch_dims=0, axis=0)
  #[512,160]
  var14077=tf.multiply(var12388, var14076)
  #[512,5,32]
  var14078=tf.reshape(var14077, [512,5,32])
  #[512,1,32]
  var14079=var14078[:,4:5]
  #[512,32]
  var14080=tf.reshape(var14079, [512,32])
  #[512,32]
  var14081=tf.cos(var14080)
  #[512,32,1]
  var14082=tf.reshape(var14081, [512,32,1])
  #[512,32]
  var14083=tf.sin(var14080)
  #[512,32]
  var14084=tf.negative(var14083)
  #[512,32,1]
  var14085=tf.reshape(var14084, [512,32,1])
  #[512,32,2]
  var14086=tf.concat([var14082,var14085], axis=2)
  #[512,32,1,2]
  var14087=tf.reshape(var14086, [512,32,1,2])
  #[512,32,1]
  var14088=tf.reshape(var14083, [512,32,1])
  #[512,32,2]
  var14089=tf.concat([var14088,var14082], axis=2)
  #[512,32,1,2]
  var14090=tf.reshape(var14089, [512,32,1,2])
  #[512,32,2,2]
  var14091=tf.concat([var14087,var14090], axis=2)
  #[512,32,1,2]
  var14092=tf.matmul(var14073, var14091)
  #[512,64]
  var14093=tf.reshape(var14092, [512,64])
  #[512,65]
  var14094=tf.concat([var14071,var14093], axis=1)
  #[512,64]
  var14095=var14094[:,0:64]
  #[512,32,1,2]
  var14096=tf.reshape(var14095, [512,32,1,2])
  #[512,1,32]
  var14097=var14078[:,3:4]
  #[512,32]
  var14098=tf.reshape(var14097, [512,32])
  #[512,32]
  var14099=tf.cos(var14098)
  #[512,32,1]
  var14100=tf.reshape(var14099, [512,32,1])
  #[512,32]
  var14101=tf.sin(var14098)
  #[512,32]
  var14102=tf.negative(var14101)
  #[512,32,1]
  var14103=tf.reshape(var14102, [512,32,1])
  #[512,32,2]
  var14104=tf.concat([var14100,var14103], axis=2)
  #[512,32,1,2]
  var14105=tf.reshape(var14104, [512,32,1,2])
  #[512,32,1]
  var14106=tf.reshape(var14101, [512,32,1])
  #[512,32,2]
  var14107=tf.concat([var14106,var14100], axis=2)
  #[512,32,1,2]
  var14108=tf.reshape(var14107, [512,32,1,2])
  #[512,32,2,2]
  var14109=tf.concat([var14105,var14108], axis=2)
  #[512,32,1,2]
  var14110=tf.matmul(var14096, var14109)
  #[512,64]
  var14111=tf.reshape(var14110, [512,64])
  #[512,1]
  var14112=var14094[:,64:65]
  #[512,1]
  var14113=tf.reshape(var14112, [512,1])
  #[512,65]
  var14114=tf.concat([var14111,var14113], axis=1)
  #[512,1]
  var14115=var14114[:,0:1]
  #[512,1]
  var14116=tf.reshape(var14115, [512,1])
  #[512,64]
  var14117=var14114[:,1:65]
  #[512,32,1,2]
  var14118=tf.reshape(var14117, [512,32,1,2])
  #[512,1,32]
  var14119=var14078[:,2:3]
  #[512,32]
  var14120=tf.reshape(var14119, [512,32])
  #[512,32]
  var14121=tf.cos(var14120)
  #[512,32,1]
  var14122=tf.reshape(var14121, [512,32,1])
  #[512,32]
  var14123=tf.sin(var14120)
  #[512,32]
  var14124=tf.negative(var14123)
  #[512,32,1]
  var14125=tf.reshape(var14124, [512,32,1])
  #[512,32,2]
  var14126=tf.concat([var14122,var14125], axis=2)
  #[512,32,1,2]
  var14127=tf.reshape(var14126, [512,32,1,2])
  #[512,32,1]
  var14128=tf.reshape(var14123, [512,32,1])
  #[512,32,2]
  var14129=tf.concat([var14128,var14122], axis=2)
  #[512,32,1,2]
  var14130=tf.reshape(var14129, [512,32,1,2])
  #[512,32,2,2]
  var14131=tf.concat([var14127,var14130], axis=2)
  #[512,32,1,2]
  var14132=tf.matmul(var14118, var14131)
  #[512,64]
  var14133=tf.reshape(var14132, [512,64])
  #[512,65]
  var14134=tf.concat([var14116,var14133], axis=1)
  #[512,64]
  var14135=var14134[:,0:64]
  #[512,32,1,2]
  var14136=tf.reshape(var14135, [512,32,1,2])
  #[512,1,32]
  var14137=var14078[:,1:2]
  #[512,32]
  var14138=tf.reshape(var14137, [512,32])
  #[512,32]
  var14139=tf.cos(var14138)
  #[512,32,1]
  var14140=tf.reshape(var14139, [512,32,1])
  #[512,32]
  var14141=tf.sin(var14138)
  #[512,32]
  var14142=tf.negative(var14141)
  #[512,32,1]
  var14143=tf.reshape(var14142, [512,32,1])
  #[512,32,2]
  var14144=tf.concat([var14140,var14143], axis=2)
  #[512,32,1,2]
  var14145=tf.reshape(var14144, [512,32,1,2])
  #[512,32,1]
  var14146=tf.reshape(var14141, [512,32,1])
  #[512,32,2]
  var14147=tf.concat([var14146,var14140], axis=2)
  #[512,32,1,2]
  var14148=tf.reshape(var14147, [512,32,1,2])
  #[512,32,2,2]
  var14149=tf.concat([var14145,var14148], axis=2)
  #[512,32,1,2]
  var14150=tf.matmul(var14136, var14149)
  #[512,64]
  var14151=tf.reshape(var14150, [512,64])
  #[512,1]
  var14152=var14134[:,64:65]
  #[512,1]
  var14153=tf.reshape(var14152, [512,1])
  #[512,65]
  var14154=tf.concat([var14151,var14153], axis=1)
  #[512,1]
  var14155=var14154[:,0:1]
  #[512,1]
  var14156=tf.reshape(var14155, [512,1])
  #[512,64]
  var14157=var14154[:,1:65]
  #[512,32,1,2]
  var14158=tf.reshape(var14157, [512,32,1,2])
  #[512,1,32]
  var14159=var14078[:,0:1]
  #[512,32]
  var14160=tf.reshape(var14159, [512,32])
  #[512,32]
  var14161=tf.cos(var14160)
  #[512,32,1]
  var14162=tf.reshape(var14161, [512,32,1])
  #[512,32]
  var14163=tf.sin(var14160)
  #[512,32]
  var14164=tf.negative(var14163)
  #[512,32,1]
  var14165=tf.reshape(var14164, [512,32,1])
  #[512,32,2]
  var14166=tf.concat([var14162,var14165], axis=2)
  #[512,32,1,2]
  var14167=tf.reshape(var14166, [512,32,1,2])
  #[512,32,1]
  var14168=tf.reshape(var14163, [512,32,1])
  #[512,32,2]
  var14169=tf.concat([var14168,var14162], axis=2)
  #[512,32,1,2]
  var14170=tf.reshape(var14169, [512,32,1,2])
  #[512,32,2,2]
  var14171=tf.concat([var14167,var14170], axis=2)
  #[512,32,1,2]
  var14172=tf.matmul(var14158, var14171)
  #[512,64]
  var14173=tf.reshape(var14172, [512,64])
  #[512,65]
  var14174=tf.concat([var14156,var14173], axis=1)
  #[512,65]
  var14175=tf.multiply(var12364, var14174)
  #[512,65]
  var14176=tf.reshape(var14175, [512,65])
  #[512,12]
  var14177=tf.matmul(var14176, var12494)
  #[512,12]
  var14178=tf.reshape(var14177, [512,12])
  #[512,12]
  var14179=tf.add(var14178, var12498)
  #[512,1,12]
  var14180=tf.reshape(var14179, [512,1,12])
  #[512,65]
  var14181=tf.multiply(var12368, var14174)
  #[512,1]
  var14182=var14181[:,0:1]
  #[512,1]
  var14183=tf.reshape(var14182, [512,1])
  #[512,64]
  var14184=var14181[:,1:65]
  #[512,32,1,2]
  var14185=tf.reshape(var14184, [512,32,1,2])
  #[512,1]
  var14186=var12390[:,16:17]
  #[512]
  var14187=tf.reshape(var14186, [512])
  #[512,160]
  var14188=tf.gather(params=var12389, indices=var14187, batch_dims=0, axis=0)
  #[512,160]
  var14189=tf.multiply(var12388, var14188)
  #[512,5,32]
  var14190=tf.reshape(var14189, [512,5,32])
  #[512,1,32]
  var14191=var14190[:,4:5]
  #[512,32]
  var14192=tf.reshape(var14191, [512,32])
  #[512,32]
  var14193=tf.cos(var14192)
  #[512,32,1]
  var14194=tf.reshape(var14193, [512,32,1])
  #[512,32]
  var14195=tf.sin(var14192)
  #[512,32]
  var14196=tf.negative(var14195)
  #[512,32,1]
  var14197=tf.reshape(var14196, [512,32,1])
  #[512,32,2]
  var14198=tf.concat([var14194,var14197], axis=2)
  #[512,32,1,2]
  var14199=tf.reshape(var14198, [512,32,1,2])
  #[512,32,1]
  var14200=tf.reshape(var14195, [512,32,1])
  #[512,32,2]
  var14201=tf.concat([var14200,var14194], axis=2)
  #[512,32,1,2]
  var14202=tf.reshape(var14201, [512,32,1,2])
  #[512,32,2,2]
  var14203=tf.concat([var14199,var14202], axis=2)
  #[512,32,1,2]
  var14204=tf.matmul(var14185, var14203)
  #[512,64]
  var14205=tf.reshape(var14204, [512,64])
  #[512,65]
  var14206=tf.concat([var14183,var14205], axis=1)
  #[512,64]
  var14207=var14206[:,0:64]
  #[512,32,1,2]
  var14208=tf.reshape(var14207, [512,32,1,2])
  #[512,1,32]
  var14209=var14190[:,3:4]
  #[512,32]
  var14210=tf.reshape(var14209, [512,32])
  #[512,32]
  var14211=tf.cos(var14210)
  #[512,32,1]
  var14212=tf.reshape(var14211, [512,32,1])
  #[512,32]
  var14213=tf.sin(var14210)
  #[512,32]
  var14214=tf.negative(var14213)
  #[512,32,1]
  var14215=tf.reshape(var14214, [512,32,1])
  #[512,32,2]
  var14216=tf.concat([var14212,var14215], axis=2)
  #[512,32,1,2]
  var14217=tf.reshape(var14216, [512,32,1,2])
  #[512,32,1]
  var14218=tf.reshape(var14213, [512,32,1])
  #[512,32,2]
  var14219=tf.concat([var14218,var14212], axis=2)
  #[512,32,1,2]
  var14220=tf.reshape(var14219, [512,32,1,2])
  #[512,32,2,2]
  var14221=tf.concat([var14217,var14220], axis=2)
  #[512,32,1,2]
  var14222=tf.matmul(var14208, var14221)
  #[512,64]
  var14223=tf.reshape(var14222, [512,64])
  #[512,1]
  var14224=var14206[:,64:65]
  #[512,1]
  var14225=tf.reshape(var14224, [512,1])
  #[512,65]
  var14226=tf.concat([var14223,var14225], axis=1)
  #[512,1]
  var14227=var14226[:,0:1]
  #[512,1]
  var14228=tf.reshape(var14227, [512,1])
  #[512,64]
  var14229=var14226[:,1:65]
  #[512,32,1,2]
  var14230=tf.reshape(var14229, [512,32,1,2])
  #[512,1,32]
  var14231=var14190[:,2:3]
  #[512,32]
  var14232=tf.reshape(var14231, [512,32])
  #[512,32]
  var14233=tf.cos(var14232)
  #[512,32,1]
  var14234=tf.reshape(var14233, [512,32,1])
  #[512,32]
  var14235=tf.sin(var14232)
  #[512,32]
  var14236=tf.negative(var14235)
  #[512,32,1]
  var14237=tf.reshape(var14236, [512,32,1])
  #[512,32,2]
  var14238=tf.concat([var14234,var14237], axis=2)
  #[512,32,1,2]
  var14239=tf.reshape(var14238, [512,32,1,2])
  #[512,32,1]
  var14240=tf.reshape(var14235, [512,32,1])
  #[512,32,2]
  var14241=tf.concat([var14240,var14234], axis=2)
  #[512,32,1,2]
  var14242=tf.reshape(var14241, [512,32,1,2])
  #[512,32,2,2]
  var14243=tf.concat([var14239,var14242], axis=2)
  #[512,32,1,2]
  var14244=tf.matmul(var14230, var14243)
  #[512,64]
  var14245=tf.reshape(var14244, [512,64])
  #[512,65]
  var14246=tf.concat([var14228,var14245], axis=1)
  #[512,64]
  var14247=var14246[:,0:64]
  #[512,32,1,2]
  var14248=tf.reshape(var14247, [512,32,1,2])
  #[512,1,32]
  var14249=var14190[:,1:2]
  #[512,32]
  var14250=tf.reshape(var14249, [512,32])
  #[512,32]
  var14251=tf.cos(var14250)
  #[512,32,1]
  var14252=tf.reshape(var14251, [512,32,1])
  #[512,32]
  var14253=tf.sin(var14250)
  #[512,32]
  var14254=tf.negative(var14253)
  #[512,32,1]
  var14255=tf.reshape(var14254, [512,32,1])
  #[512,32,2]
  var14256=tf.concat([var14252,var14255], axis=2)
  #[512,32,1,2]
  var14257=tf.reshape(var14256, [512,32,1,2])
  #[512,32,1]
  var14258=tf.reshape(var14253, [512,32,1])
  #[512,32,2]
  var14259=tf.concat([var14258,var14252], axis=2)
  #[512,32,1,2]
  var14260=tf.reshape(var14259, [512,32,1,2])
  #[512,32,2,2]
  var14261=tf.concat([var14257,var14260], axis=2)
  #[512,32,1,2]
  var14262=tf.matmul(var14248, var14261)
  #[512,64]
  var14263=tf.reshape(var14262, [512,64])
  #[512,1]
  var14264=var14246[:,64:65]
  #[512,1]
  var14265=tf.reshape(var14264, [512,1])
  #[512,65]
  var14266=tf.concat([var14263,var14265], axis=1)
  #[512,1]
  var14267=var14266[:,0:1]
  #[512,1]
  var14268=tf.reshape(var14267, [512,1])
  #[512,64]
  var14269=var14266[:,1:65]
  #[512,32,1,2]
  var14270=tf.reshape(var14269, [512,32,1,2])
  #[512,1,32]
  var14271=var14190[:,0:1]
  #[512,32]
  var14272=tf.reshape(var14271, [512,32])
  #[512,32]
  var14273=tf.cos(var14272)
  #[512,32,1]
  var14274=tf.reshape(var14273, [512,32,1])
  #[512,32]
  var14275=tf.sin(var14272)
  #[512,32]
  var14276=tf.negative(var14275)
  #[512,32,1]
  var14277=tf.reshape(var14276, [512,32,1])
  #[512,32,2]
  var14278=tf.concat([var14274,var14277], axis=2)
  #[512,32,1,2]
  var14279=tf.reshape(var14278, [512,32,1,2])
  #[512,32,1]
  var14280=tf.reshape(var14275, [512,32,1])
  #[512,32,2]
  var14281=tf.concat([var14280,var14274], axis=2)
  #[512,32,1,2]
  var14282=tf.reshape(var14281, [512,32,1,2])
  #[512,32,2,2]
  var14283=tf.concat([var14279,var14282], axis=2)
  #[512,32,1,2]
  var14284=tf.matmul(var14270, var14283)
  #[512,64]
  var14285=tf.reshape(var14284, [512,64])
  #[512,65]
  var14286=tf.concat([var14268,var14285], axis=1)
  #[512,65]
  var14287=tf.multiply(var12364, var14286)
  #[512,65]
  var14288=tf.reshape(var14287, [512,65])
  #[512,12]
  var14289=tf.matmul(var14288, var12494)
  #[512,12]
  var14290=tf.reshape(var14289, [512,12])
  #[512,12]
  var14291=tf.add(var14290, var12498)
  #[512,1,12]
  var14292=tf.reshape(var14291, [512,1,12])
  #[512,65]
  var14293=tf.multiply(var12368, var14286)
  #[512,1]
  var14294=var14293[:,0:1]
  #[512,1]
  var14295=tf.reshape(var14294, [512,1])
  #[512,64]
  var14296=var14293[:,1:65]
  #[512,32,1,2]
  var14297=tf.reshape(var14296, [512,32,1,2])
  #[512,1]
  var14298=var12390[:,17:18]
  #[512]
  var14299=tf.reshape(var14298, [512])
  #[512,160]
  var14300=tf.gather(params=var12389, indices=var14299, batch_dims=0, axis=0)
  #[512,160]
  var14301=tf.multiply(var12388, var14300)
  #[512,5,32]
  var14302=tf.reshape(var14301, [512,5,32])
  #[512,1,32]
  var14303=var14302[:,4:5]
  #[512,32]
  var14304=tf.reshape(var14303, [512,32])
  #[512,32]
  var14305=tf.cos(var14304)
  #[512,32,1]
  var14306=tf.reshape(var14305, [512,32,1])
  #[512,32]
  var14307=tf.sin(var14304)
  #[512,32]
  var14308=tf.negative(var14307)
  #[512,32,1]
  var14309=tf.reshape(var14308, [512,32,1])
  #[512,32,2]
  var14310=tf.concat([var14306,var14309], axis=2)
  #[512,32,1,2]
  var14311=tf.reshape(var14310, [512,32,1,2])
  #[512,32,1]
  var14312=tf.reshape(var14307, [512,32,1])
  #[512,32,2]
  var14313=tf.concat([var14312,var14306], axis=2)
  #[512,32,1,2]
  var14314=tf.reshape(var14313, [512,32,1,2])
  #[512,32,2,2]
  var14315=tf.concat([var14311,var14314], axis=2)
  #[512,32,1,2]
  var14316=tf.matmul(var14297, var14315)
  #[512,64]
  var14317=tf.reshape(var14316, [512,64])
  #[512,65]
  var14318=tf.concat([var14295,var14317], axis=1)
  #[512,64]
  var14319=var14318[:,0:64]
  #[512,32,1,2]
  var14320=tf.reshape(var14319, [512,32,1,2])
  #[512,1,32]
  var14321=var14302[:,3:4]
  #[512,32]
  var14322=tf.reshape(var14321, [512,32])
  #[512,32]
  var14323=tf.cos(var14322)
  #[512,32,1]
  var14324=tf.reshape(var14323, [512,32,1])
  #[512,32]
  var14325=tf.sin(var14322)
  #[512,32]
  var14326=tf.negative(var14325)
  #[512,32,1]
  var14327=tf.reshape(var14326, [512,32,1])
  #[512,32,2]
  var14328=tf.concat([var14324,var14327], axis=2)
  #[512,32,1,2]
  var14329=tf.reshape(var14328, [512,32,1,2])
  #[512,32,1]
  var14330=tf.reshape(var14325, [512,32,1])
  #[512,32,2]
  var14331=tf.concat([var14330,var14324], axis=2)
  #[512,32,1,2]
  var14332=tf.reshape(var14331, [512,32,1,2])
  #[512,32,2,2]
  var14333=tf.concat([var14329,var14332], axis=2)
  #[512,32,1,2]
  var14334=tf.matmul(var14320, var14333)
  #[512,64]
  var14335=tf.reshape(var14334, [512,64])
  #[512,1]
  var14336=var14318[:,64:65]
  #[512,1]
  var14337=tf.reshape(var14336, [512,1])
  #[512,65]
  var14338=tf.concat([var14335,var14337], axis=1)
  #[512,1]
  var14339=var14338[:,0:1]
  #[512,1]
  var14340=tf.reshape(var14339, [512,1])
  #[512,64]
  var14341=var14338[:,1:65]
  #[512,32,1,2]
  var14342=tf.reshape(var14341, [512,32,1,2])
  #[512,1,32]
  var14343=var14302[:,2:3]
  #[512,32]
  var14344=tf.reshape(var14343, [512,32])
  #[512,32]
  var14345=tf.cos(var14344)
  #[512,32,1]
  var14346=tf.reshape(var14345, [512,32,1])
  #[512,32]
  var14347=tf.sin(var14344)
  #[512,32]
  var14348=tf.negative(var14347)
  #[512,32,1]
  var14349=tf.reshape(var14348, [512,32,1])
  #[512,32,2]
  var14350=tf.concat([var14346,var14349], axis=2)
  #[512,32,1,2]
  var14351=tf.reshape(var14350, [512,32,1,2])
  #[512,32,1]
  var14352=tf.reshape(var14347, [512,32,1])
  #[512,32,2]
  var14353=tf.concat([var14352,var14346], axis=2)
  #[512,32,1,2]
  var14354=tf.reshape(var14353, [512,32,1,2])
  #[512,32,2,2]
  var14355=tf.concat([var14351,var14354], axis=2)
  #[512,32,1,2]
  var14356=tf.matmul(var14342, var14355)
  #[512,64]
  var14357=tf.reshape(var14356, [512,64])
  #[512,65]
  var14358=tf.concat([var14340,var14357], axis=1)
  #[512,64]
  var14359=var14358[:,0:64]
  #[512,32,1,2]
  var14360=tf.reshape(var14359, [512,32,1,2])
  #[512,1,32]
  var14361=var14302[:,1:2]
  #[512,32]
  var14362=tf.reshape(var14361, [512,32])
  #[512,32]
  var14363=tf.cos(var14362)
  #[512,32,1]
  var14364=tf.reshape(var14363, [512,32,1])
  #[512,32]
  var14365=tf.sin(var14362)
  #[512,32]
  var14366=tf.negative(var14365)
  #[512,32,1]
  var14367=tf.reshape(var14366, [512,32,1])
  #[512,32,2]
  var14368=tf.concat([var14364,var14367], axis=2)
  #[512,32,1,2]
  var14369=tf.reshape(var14368, [512,32,1,2])
  #[512,32,1]
  var14370=tf.reshape(var14365, [512,32,1])
  #[512,32,2]
  var14371=tf.concat([var14370,var14364], axis=2)
  #[512,32,1,2]
  var14372=tf.reshape(var14371, [512,32,1,2])
  #[512,32,2,2]
  var14373=tf.concat([var14369,var14372], axis=2)
  #[512,32,1,2]
  var14374=tf.matmul(var14360, var14373)
  #[512,64]
  var14375=tf.reshape(var14374, [512,64])
  #[512,1]
  var14376=var14358[:,64:65]
  #[512,1]
  var14377=tf.reshape(var14376, [512,1])
  #[512,65]
  var14378=tf.concat([var14375,var14377], axis=1)
  #[512,1]
  var14379=var14378[:,0:1]
  #[512,1]
  var14380=tf.reshape(var14379, [512,1])
  #[512,64]
  var14381=var14378[:,1:65]
  #[512,32,1,2]
  var14382=tf.reshape(var14381, [512,32,1,2])
  #[512,1,32]
  var14383=var14302[:,0:1]
  #[512,32]
  var14384=tf.reshape(var14383, [512,32])
  #[512,32]
  var14385=tf.cos(var14384)
  #[512,32,1]
  var14386=tf.reshape(var14385, [512,32,1])
  #[512,32]
  var14387=tf.sin(var14384)
  #[512,32]
  var14388=tf.negative(var14387)
  #[512,32,1]
  var14389=tf.reshape(var14388, [512,32,1])
  #[512,32,2]
  var14390=tf.concat([var14386,var14389], axis=2)
  #[512,32,1,2]
  var14391=tf.reshape(var14390, [512,32,1,2])
  #[512,32,1]
  var14392=tf.reshape(var14387, [512,32,1])
  #[512,32,2]
  var14393=tf.concat([var14392,var14386], axis=2)
  #[512,32,1,2]
  var14394=tf.reshape(var14393, [512,32,1,2])
  #[512,32,2,2]
  var14395=tf.concat([var14391,var14394], axis=2)
  #[512,32,1,2]
  var14396=tf.matmul(var14382, var14395)
  #[512,64]
  var14397=tf.reshape(var14396, [512,64])
  #[512,65]
  var14398=tf.concat([var14380,var14397], axis=1)
  #[512,65]
  var14399=tf.multiply(var12364, var14398)
  #[512,65]
  var14400=tf.reshape(var14399, [512,65])
  #[512,12]
  var14401=tf.matmul(var14400, var12494)
  #[512,12]
  var14402=tf.reshape(var14401, [512,12])
  #[512,12]
  var14403=tf.add(var14402, var12498)
  #[512,1,12]
  var14404=tf.reshape(var14403, [512,1,12])
  #[512,65]
  var14405=tf.multiply(var12368, var14398)
  #[512,1]
  var14406=var14405[:,0:1]
  #[512,1]
  var14407=tf.reshape(var14406, [512,1])
  #[512,64]
  var14408=var14405[:,1:65]
  #[512,32,1,2]
  var14409=tf.reshape(var14408, [512,32,1,2])
  #[512,1]
  var14410=var12390[:,18:19]
  #[512]
  var14411=tf.reshape(var14410, [512])
  #[512,160]
  var14412=tf.gather(params=var12389, indices=var14411, batch_dims=0, axis=0)
  #[512,160]
  var14413=tf.multiply(var12388, var14412)
  #[512,5,32]
  var14414=tf.reshape(var14413, [512,5,32])
  #[512,1,32]
  var14415=var14414[:,4:5]
  #[512,32]
  var14416=tf.reshape(var14415, [512,32])
  #[512,32]
  var14417=tf.cos(var14416)
  #[512,32,1]
  var14418=tf.reshape(var14417, [512,32,1])
  #[512,32]
  var14419=tf.sin(var14416)
  #[512,32]
  var14420=tf.negative(var14419)
  #[512,32,1]
  var14421=tf.reshape(var14420, [512,32,1])
  #[512,32,2]
  var14422=tf.concat([var14418,var14421], axis=2)
  #[512,32,1,2]
  var14423=tf.reshape(var14422, [512,32,1,2])
  #[512,32,1]
  var14424=tf.reshape(var14419, [512,32,1])
  #[512,32,2]
  var14425=tf.concat([var14424,var14418], axis=2)
  #[512,32,1,2]
  var14426=tf.reshape(var14425, [512,32,1,2])
  #[512,32,2,2]
  var14427=tf.concat([var14423,var14426], axis=2)
  #[512,32,1,2]
  var14428=tf.matmul(var14409, var14427)
  #[512,64]
  var14429=tf.reshape(var14428, [512,64])
  #[512,65]
  var14430=tf.concat([var14407,var14429], axis=1)
  #[512,64]
  var14431=var14430[:,0:64]
  #[512,32,1,2]
  var14432=tf.reshape(var14431, [512,32,1,2])
  #[512,1,32]
  var14433=var14414[:,3:4]
  #[512,32]
  var14434=tf.reshape(var14433, [512,32])
  #[512,32]
  var14435=tf.cos(var14434)
  #[512,32,1]
  var14436=tf.reshape(var14435, [512,32,1])
  #[512,32]
  var14437=tf.sin(var14434)
  #[512,32]
  var14438=tf.negative(var14437)
  #[512,32,1]
  var14439=tf.reshape(var14438, [512,32,1])
  #[512,32,2]
  var14440=tf.concat([var14436,var14439], axis=2)
  #[512,32,1,2]
  var14441=tf.reshape(var14440, [512,32,1,2])
  #[512,32,1]
  var14442=tf.reshape(var14437, [512,32,1])
  #[512,32,2]
  var14443=tf.concat([var14442,var14436], axis=2)
  #[512,32,1,2]
  var14444=tf.reshape(var14443, [512,32,1,2])
  #[512,32,2,2]
  var14445=tf.concat([var14441,var14444], axis=2)
  #[512,32,1,2]
  var14446=tf.matmul(var14432, var14445)
  #[512,64]
  var14447=tf.reshape(var14446, [512,64])
  #[512,1]
  var14448=var14430[:,64:65]
  #[512,1]
  var14449=tf.reshape(var14448, [512,1])
  #[512,65]
  var14450=tf.concat([var14447,var14449], axis=1)
  #[512,1]
  var14451=var14450[:,0:1]
  #[512,1]
  var14452=tf.reshape(var14451, [512,1])
  #[512,64]
  var14453=var14450[:,1:65]
  #[512,32,1,2]
  var14454=tf.reshape(var14453, [512,32,1,2])
  #[512,1,32]
  var14455=var14414[:,2:3]
  #[512,32]
  var14456=tf.reshape(var14455, [512,32])
  #[512,32]
  var14457=tf.cos(var14456)
  #[512,32,1]
  var14458=tf.reshape(var14457, [512,32,1])
  #[512,32]
  var14459=tf.sin(var14456)
  #[512,32]
  var14460=tf.negative(var14459)
  #[512,32,1]
  var14461=tf.reshape(var14460, [512,32,1])
  #[512,32,2]
  var14462=tf.concat([var14458,var14461], axis=2)
  #[512,32,1,2]
  var14463=tf.reshape(var14462, [512,32,1,2])
  #[512,32,1]
  var14464=tf.reshape(var14459, [512,32,1])
  #[512,32,2]
  var14465=tf.concat([var14464,var14458], axis=2)
  #[512,32,1,2]
  var14466=tf.reshape(var14465, [512,32,1,2])
  #[512,32,2,2]
  var14467=tf.concat([var14463,var14466], axis=2)
  #[512,32,1,2]
  var14468=tf.matmul(var14454, var14467)
  #[512,64]
  var14469=tf.reshape(var14468, [512,64])
  #[512,65]
  var14470=tf.concat([var14452,var14469], axis=1)
  #[512,64]
  var14471=var14470[:,0:64]
  #[512,32,1,2]
  var14472=tf.reshape(var14471, [512,32,1,2])
  #[512,1,32]
  var14473=var14414[:,1:2]
  #[512,32]
  var14474=tf.reshape(var14473, [512,32])
  #[512,32]
  var14475=tf.cos(var14474)
  #[512,32,1]
  var14476=tf.reshape(var14475, [512,32,1])
  #[512,32]
  var14477=tf.sin(var14474)
  #[512,32]
  var14478=tf.negative(var14477)
  #[512,32,1]
  var14479=tf.reshape(var14478, [512,32,1])
  #[512,32,2]
  var14480=tf.concat([var14476,var14479], axis=2)
  #[512,32,1,2]
  var14481=tf.reshape(var14480, [512,32,1,2])
  #[512,32,1]
  var14482=tf.reshape(var14477, [512,32,1])
  #[512,32,2]
  var14483=tf.concat([var14482,var14476], axis=2)
  #[512,32,1,2]
  var14484=tf.reshape(var14483, [512,32,1,2])
  #[512,32,2,2]
  var14485=tf.concat([var14481,var14484], axis=2)
  #[512,32,1,2]
  var14486=tf.matmul(var14472, var14485)
  #[512,64]
  var14487=tf.reshape(var14486, [512,64])
  #[512,1]
  var14488=var14470[:,64:65]
  #[512,1]
  var14489=tf.reshape(var14488, [512,1])
  #[512,65]
  var14490=tf.concat([var14487,var14489], axis=1)
  #[512,1]
  var14491=var14490[:,0:1]
  #[512,1]
  var14492=tf.reshape(var14491, [512,1])
  #[512,64]
  var14493=var14490[:,1:65]
  #[512,32,1,2]
  var14494=tf.reshape(var14493, [512,32,1,2])
  #[512,1,32]
  var14495=var14414[:,0:1]
  #[512,32]
  var14496=tf.reshape(var14495, [512,32])
  #[512,32]
  var14497=tf.cos(var14496)
  #[512,32,1]
  var14498=tf.reshape(var14497, [512,32,1])
  #[512,32]
  var14499=tf.sin(var14496)
  #[512,32]
  var14500=tf.negative(var14499)
  #[512,32,1]
  var14501=tf.reshape(var14500, [512,32,1])
  #[512,32,2]
  var14502=tf.concat([var14498,var14501], axis=2)
  #[512,32,1,2]
  var14503=tf.reshape(var14502, [512,32,1,2])
  #[512,32,1]
  var14504=tf.reshape(var14499, [512,32,1])
  #[512,32,2]
  var14505=tf.concat([var14504,var14498], axis=2)
  #[512,32,1,2]
  var14506=tf.reshape(var14505, [512,32,1,2])
  #[512,32,2,2]
  var14507=tf.concat([var14503,var14506], axis=2)
  #[512,32,1,2]
  var14508=tf.matmul(var14494, var14507)
  #[512,64]
  var14509=tf.reshape(var14508, [512,64])
  #[512,65]
  var14510=tf.concat([var14492,var14509], axis=1)
  #[512,65]
  var14511=tf.multiply(var12364, var14510)
  #[512,65]
  var14512=tf.reshape(var14511, [512,65])
  #[512,12]
  var14513=tf.matmul(var14512, var12494)
  #[512,12]
  var14514=tf.reshape(var14513, [512,12])
  #[512,12]
  var14515=tf.add(var14514, var12498)
  #[512,1,12]
  var14516=tf.reshape(var14515, [512,1,12])
  #[512,65]
  var14517=tf.multiply(var12368, var14510)
  #[512,1]
  var14518=var14517[:,0:1]
  #[512,1]
  var14519=tf.reshape(var14518, [512,1])
  #[512,64]
  var14520=var14517[:,1:65]
  #[512,32,1,2]
  var14521=tf.reshape(var14520, [512,32,1,2])
  #[512,1]
  var14522=var12390[:,19:20]
  #[512]
  var14523=tf.reshape(var14522, [512])
  #[512,160]
  var14524=tf.gather(params=var12389, indices=var14523, batch_dims=0, axis=0)
  #[512,160]
  var14525=tf.multiply(var12388, var14524)
  #[512,5,32]
  var14526=tf.reshape(var14525, [512,5,32])
  #[512,1,32]
  var14527=var14526[:,4:5]
  #[512,32]
  var14528=tf.reshape(var14527, [512,32])
  #[512,32]
  var14529=tf.cos(var14528)
  #[512,32,1]
  var14530=tf.reshape(var14529, [512,32,1])
  #[512,32]
  var14531=tf.sin(var14528)
  #[512,32]
  var14532=tf.negative(var14531)
  #[512,32,1]
  var14533=tf.reshape(var14532, [512,32,1])
  #[512,32,2]
  var14534=tf.concat([var14530,var14533], axis=2)
  #[512,32,1,2]
  var14535=tf.reshape(var14534, [512,32,1,2])
  #[512,32,1]
  var14536=tf.reshape(var14531, [512,32,1])
  #[512,32,2]
  var14537=tf.concat([var14536,var14530], axis=2)
  #[512,32,1,2]
  var14538=tf.reshape(var14537, [512,32,1,2])
  #[512,32,2,2]
  var14539=tf.concat([var14535,var14538], axis=2)
  #[512,32,1,2]
  var14540=tf.matmul(var14521, var14539)
  #[512,64]
  var14541=tf.reshape(var14540, [512,64])
  #[512,65]
  var14542=tf.concat([var14519,var14541], axis=1)
  #[512,64]
  var14543=var14542[:,0:64]
  #[512,32,1,2]
  var14544=tf.reshape(var14543, [512,32,1,2])
  #[512,1,32]
  var14545=var14526[:,3:4]
  #[512,32]
  var14546=tf.reshape(var14545, [512,32])
  #[512,32]
  var14547=tf.cos(var14546)
  #[512,32,1]
  var14548=tf.reshape(var14547, [512,32,1])
  #[512,32]
  var14549=tf.sin(var14546)
  #[512,32]
  var14550=tf.negative(var14549)
  #[512,32,1]
  var14551=tf.reshape(var14550, [512,32,1])
  #[512,32,2]
  var14552=tf.concat([var14548,var14551], axis=2)
  #[512,32,1,2]
  var14553=tf.reshape(var14552, [512,32,1,2])
  #[512,32,1]
  var14554=tf.reshape(var14549, [512,32,1])
  #[512,32,2]
  var14555=tf.concat([var14554,var14548], axis=2)
  #[512,32,1,2]
  var14556=tf.reshape(var14555, [512,32,1,2])
  #[512,32,2,2]
  var14557=tf.concat([var14553,var14556], axis=2)
  #[512,32,1,2]
  var14558=tf.matmul(var14544, var14557)
  #[512,64]
  var14559=tf.reshape(var14558, [512,64])
  #[512,1]
  var14560=var14542[:,64:65]
  #[512,1]
  var14561=tf.reshape(var14560, [512,1])
  #[512,65]
  var14562=tf.concat([var14559,var14561], axis=1)
  #[512,1]
  var14563=var14562[:,0:1]
  #[512,1]
  var14564=tf.reshape(var14563, [512,1])
  #[512,64]
  var14565=var14562[:,1:65]
  #[512,32,1,2]
  var14566=tf.reshape(var14565, [512,32,1,2])
  #[512,1,32]
  var14567=var14526[:,2:3]
  #[512,32]
  var14568=tf.reshape(var14567, [512,32])
  #[512,32]
  var14569=tf.cos(var14568)
  #[512,32,1]
  var14570=tf.reshape(var14569, [512,32,1])
  #[512,32]
  var14571=tf.sin(var14568)
  #[512,32]
  var14572=tf.negative(var14571)
  #[512,32,1]
  var14573=tf.reshape(var14572, [512,32,1])
  #[512,32,2]
  var14574=tf.concat([var14570,var14573], axis=2)
  #[512,32,1,2]
  var14575=tf.reshape(var14574, [512,32,1,2])
  #[512,32,1]
  var14576=tf.reshape(var14571, [512,32,1])
  #[512,32,2]
  var14577=tf.concat([var14576,var14570], axis=2)
  #[512,32,1,2]
  var14578=tf.reshape(var14577, [512,32,1,2])
  #[512,32,2,2]
  var14579=tf.concat([var14575,var14578], axis=2)
  #[512,32,1,2]
  var14580=tf.matmul(var14566, var14579)
  #[512,64]
  var14581=tf.reshape(var14580, [512,64])
  #[512,65]
  var14582=tf.concat([var14564,var14581], axis=1)
  #[512,64]
  var14583=var14582[:,0:64]
  #[512,32,1,2]
  var14584=tf.reshape(var14583, [512,32,1,2])
  #[512,1,32]
  var14585=var14526[:,1:2]
  #[512,32]
  var14586=tf.reshape(var14585, [512,32])
  #[512,32]
  var14587=tf.cos(var14586)
  #[512,32,1]
  var14588=tf.reshape(var14587, [512,32,1])
  #[512,32]
  var14589=tf.sin(var14586)
  #[512,32]
  var14590=tf.negative(var14589)
  #[512,32,1]
  var14591=tf.reshape(var14590, [512,32,1])
  #[512,32,2]
  var14592=tf.concat([var14588,var14591], axis=2)
  #[512,32,1,2]
  var14593=tf.reshape(var14592, [512,32,1,2])
  #[512,32,1]
  var14594=tf.reshape(var14589, [512,32,1])
  #[512,32,2]
  var14595=tf.concat([var14594,var14588], axis=2)
  #[512,32,1,2]
  var14596=tf.reshape(var14595, [512,32,1,2])
  #[512,32,2,2]
  var14597=tf.concat([var14593,var14596], axis=2)
  #[512,32,1,2]
  var14598=tf.matmul(var14584, var14597)
  #[512,64]
  var14599=tf.reshape(var14598, [512,64])
  #[512,1]
  var14600=var14582[:,64:65]
  #[512,1]
  var14601=tf.reshape(var14600, [512,1])
  #[512,65]
  var14602=tf.concat([var14599,var14601], axis=1)
  #[512,1]
  var14603=var14602[:,0:1]
  #[512,1]
  var14604=tf.reshape(var14603, [512,1])
  #[512,64]
  var14605=var14602[:,1:65]
  #[512,32,1,2]
  var14606=tf.reshape(var14605, [512,32,1,2])
  #[512,1,32]
  var14607=var14526[:,0:1]
  #[512,32]
  var14608=tf.reshape(var14607, [512,32])
  #[512,32]
  var14609=tf.cos(var14608)
  #[512,32,1]
  var14610=tf.reshape(var14609, [512,32,1])
  #[512,32]
  var14611=tf.sin(var14608)
  #[512,32]
  var14612=tf.negative(var14611)
  #[512,32,1]
  var14613=tf.reshape(var14612, [512,32,1])
  #[512,32,2]
  var14614=tf.concat([var14610,var14613], axis=2)
  #[512,32,1,2]
  var14615=tf.reshape(var14614, [512,32,1,2])
  #[512,32,1]
  var14616=tf.reshape(var14611, [512,32,1])
  #[512,32,2]
  var14617=tf.concat([var14616,var14610], axis=2)
  #[512,32,1,2]
  var14618=tf.reshape(var14617, [512,32,1,2])
  #[512,32,2,2]
  var14619=tf.concat([var14615,var14618], axis=2)
  #[512,32,1,2]
  var14620=tf.matmul(var14606, var14619)
  #[512,64]
  var14621=tf.reshape(var14620, [512,64])
  #[512,65]
  var14622=tf.concat([var14604,var14621], axis=1)
  #[512,65]
  var14623=tf.multiply(var12364, var14622)
  #[512,65]
  var14624=tf.reshape(var14623, [512,65])
  #[512,12]
  var14625=tf.matmul(var14624, var12494)
  #[512,12]
  var14626=tf.reshape(var14625, [512,12])
  #[512,12]
  var14627=tf.add(var14626, var12498)
  #[512,1,12]
  var14628=tf.reshape(var14627, [512,1,12])
  #[512,65]
  var14629=tf.multiply(var12368, var14622)
  #[512,1]
  var14630=var14629[:,0:1]
  #[512,1]
  var14631=tf.reshape(var14630, [512,1])
  #[512,64]
  var14632=var14629[:,1:65]
  #[512,32,1,2]
  var14633=tf.reshape(var14632, [512,32,1,2])
  #[512,1]
  var14634=var12390[:,20:21]
  #[512]
  var14635=tf.reshape(var14634, [512])
  #[512,160]
  var14636=tf.gather(params=var12389, indices=var14635, batch_dims=0, axis=0)
  #[512,160]
  var14637=tf.multiply(var12388, var14636)
  #[512,5,32]
  var14638=tf.reshape(var14637, [512,5,32])
  #[512,1,32]
  var14639=var14638[:,4:5]
  #[512,32]
  var14640=tf.reshape(var14639, [512,32])
  #[512,32]
  var14641=tf.cos(var14640)
  #[512,32,1]
  var14642=tf.reshape(var14641, [512,32,1])
  #[512,32]
  var14643=tf.sin(var14640)
  #[512,32]
  var14644=tf.negative(var14643)
  #[512,32,1]
  var14645=tf.reshape(var14644, [512,32,1])
  #[512,32,2]
  var14646=tf.concat([var14642,var14645], axis=2)
  #[512,32,1,2]
  var14647=tf.reshape(var14646, [512,32,1,2])
  #[512,32,1]
  var14648=tf.reshape(var14643, [512,32,1])
  #[512,32,2]
  var14649=tf.concat([var14648,var14642], axis=2)
  #[512,32,1,2]
  var14650=tf.reshape(var14649, [512,32,1,2])
  #[512,32,2,2]
  var14651=tf.concat([var14647,var14650], axis=2)
  #[512,32,1,2]
  var14652=tf.matmul(var14633, var14651)
  #[512,64]
  var14653=tf.reshape(var14652, [512,64])
  #[512,65]
  var14654=tf.concat([var14631,var14653], axis=1)
  #[512,64]
  var14655=var14654[:,0:64]
  #[512,32,1,2]
  var14656=tf.reshape(var14655, [512,32,1,2])
  #[512,1,32]
  var14657=var14638[:,3:4]
  #[512,32]
  var14658=tf.reshape(var14657, [512,32])
  #[512,32]
  var14659=tf.cos(var14658)
  #[512,32,1]
  var14660=tf.reshape(var14659, [512,32,1])
  #[512,32]
  var14661=tf.sin(var14658)
  #[512,32]
  var14662=tf.negative(var14661)
  #[512,32,1]
  var14663=tf.reshape(var14662, [512,32,1])
  #[512,32,2]
  var14664=tf.concat([var14660,var14663], axis=2)
  #[512,32,1,2]
  var14665=tf.reshape(var14664, [512,32,1,2])
  #[512,32,1]
  var14666=tf.reshape(var14661, [512,32,1])
  #[512,32,2]
  var14667=tf.concat([var14666,var14660], axis=2)
  #[512,32,1,2]
  var14668=tf.reshape(var14667, [512,32,1,2])
  #[512,32,2,2]
  var14669=tf.concat([var14665,var14668], axis=2)
  #[512,32,1,2]
  var14670=tf.matmul(var14656, var14669)
  #[512,64]
  var14671=tf.reshape(var14670, [512,64])
  #[512,1]
  var14672=var14654[:,64:65]
  #[512,1]
  var14673=tf.reshape(var14672, [512,1])
  #[512,65]
  var14674=tf.concat([var14671,var14673], axis=1)
  #[512,1]
  var14675=var14674[:,0:1]
  #[512,1]
  var14676=tf.reshape(var14675, [512,1])
  #[512,64]
  var14677=var14674[:,1:65]
  #[512,32,1,2]
  var14678=tf.reshape(var14677, [512,32,1,2])
  #[512,1,32]
  var14679=var14638[:,2:3]
  #[512,32]
  var14680=tf.reshape(var14679, [512,32])
  #[512,32]
  var14681=tf.cos(var14680)
  #[512,32,1]
  var14682=tf.reshape(var14681, [512,32,1])
  #[512,32]
  var14683=tf.sin(var14680)
  #[512,32]
  var14684=tf.negative(var14683)
  #[512,32,1]
  var14685=tf.reshape(var14684, [512,32,1])
  #[512,32,2]
  var14686=tf.concat([var14682,var14685], axis=2)
  #[512,32,1,2]
  var14687=tf.reshape(var14686, [512,32,1,2])
  #[512,32,1]
  var14688=tf.reshape(var14683, [512,32,1])
  #[512,32,2]
  var14689=tf.concat([var14688,var14682], axis=2)
  #[512,32,1,2]
  var14690=tf.reshape(var14689, [512,32,1,2])
  #[512,32,2,2]
  var14691=tf.concat([var14687,var14690], axis=2)
  #[512,32,1,2]
  var14692=tf.matmul(var14678, var14691)
  #[512,64]
  var14693=tf.reshape(var14692, [512,64])
  #[512,65]
  var14694=tf.concat([var14676,var14693], axis=1)
  #[512,64]
  var14695=var14694[:,0:64]
  #[512,32,1,2]
  var14696=tf.reshape(var14695, [512,32,1,2])
  #[512,1,32]
  var14697=var14638[:,1:2]
  #[512,32]
  var14698=tf.reshape(var14697, [512,32])
  #[512,32]
  var14699=tf.cos(var14698)
  #[512,32,1]
  var14700=tf.reshape(var14699, [512,32,1])
  #[512,32]
  var14701=tf.sin(var14698)
  #[512,32]
  var14702=tf.negative(var14701)
  #[512,32,1]
  var14703=tf.reshape(var14702, [512,32,1])
  #[512,32,2]
  var14704=tf.concat([var14700,var14703], axis=2)
  #[512,32,1,2]
  var14705=tf.reshape(var14704, [512,32,1,2])
  #[512,32,1]
  var14706=tf.reshape(var14701, [512,32,1])
  #[512,32,2]
  var14707=tf.concat([var14706,var14700], axis=2)
  #[512,32,1,2]
  var14708=tf.reshape(var14707, [512,32,1,2])
  #[512,32,2,2]
  var14709=tf.concat([var14705,var14708], axis=2)
  #[512,32,1,2]
  var14710=tf.matmul(var14696, var14709)
  #[512,64]
  var14711=tf.reshape(var14710, [512,64])
  #[512,1]
  var14712=var14694[:,64:65]
  #[512,1]
  var14713=tf.reshape(var14712, [512,1])
  #[512,65]
  var14714=tf.concat([var14711,var14713], axis=1)
  #[512,1]
  var14715=var14714[:,0:1]
  #[512,1]
  var14716=tf.reshape(var14715, [512,1])
  #[512,64]
  var14717=var14714[:,1:65]
  #[512,32,1,2]
  var14718=tf.reshape(var14717, [512,32,1,2])
  #[512,1,32]
  var14719=var14638[:,0:1]
  #[512,32]
  var14720=tf.reshape(var14719, [512,32])
  #[512,32]
  var14721=tf.cos(var14720)
  #[512,32,1]
  var14722=tf.reshape(var14721, [512,32,1])
  #[512,32]
  var14723=tf.sin(var14720)
  #[512,32]
  var14724=tf.negative(var14723)
  #[512,32,1]
  var14725=tf.reshape(var14724, [512,32,1])
  #[512,32,2]
  var14726=tf.concat([var14722,var14725], axis=2)
  #[512,32,1,2]
  var14727=tf.reshape(var14726, [512,32,1,2])
  #[512,32,1]
  var14728=tf.reshape(var14723, [512,32,1])
  #[512,32,2]
  var14729=tf.concat([var14728,var14722], axis=2)
  #[512,32,1,2]
  var14730=tf.reshape(var14729, [512,32,1,2])
  #[512,32,2,2]
  var14731=tf.concat([var14727,var14730], axis=2)
  #[512,32,1,2]
  var14732=tf.matmul(var14718, var14731)
  #[512,64]
  var14733=tf.reshape(var14732, [512,64])
  #[512,65]
  var14734=tf.concat([var14716,var14733], axis=1)
  #[512,65]
  var14735=tf.multiply(var12364, var14734)
  #[512,65]
  var14736=tf.reshape(var14735, [512,65])
  #[512,12]
  var14737=tf.matmul(var14736, var12494)
  #[512,12]
  var14738=tf.reshape(var14737, [512,12])
  #[512,12]
  var14739=tf.add(var14738, var12498)
  #[512,1,12]
  var14740=tf.reshape(var14739, [512,1,12])
  #[512,21,12]
  var14741=tf.concat([var12500
                     ,var12612
                     ,var12724
                     ,var12836
                     ,var12948
                     ,var13060
                     ,var13172
                     ,var13284
                     ,var13396
                     ,var13508
                     ,var13620
                     ,var13732
                     ,var13844
                     ,var13956
                     ,var14068
                     ,var14180
                     ,var14292
                     ,var14404
                     ,var14516
                     ,var14628
                     ,var14740],
                     axis=1)
  #[512,21]
  var14742=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=var12351, logits=var14741)
  #[512,21]
  var14743=weights
  #[512,21]
  var14744=tf.multiply(var14742, var14743)
  #[512,21]
  var14745=tf.reshape(var14744, [512,21])
  #[512]
  var14746=tf.reduce_sum(var14745, axis=1)
  #[512,21]
  var14747=tf.reshape(var14743, [512,21])
  #[512]
  var14748=tf.reduce_sum(var14747, axis=1)
  #[512]
  var14749=tf.divide(var14746, var14748)
  #[512]
  var14750=tf.cast(var14749, tf.float32)
  #[512]
  var14751=tf.reshape(var14750, [512])
  #[]
  var14752=tf.reduce_mean(var14751, axis=0)
  #[]
  var14753=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var14754=tf.broadcast_to(tf.reshape(var14753, [1]), [1])
  #[]
  var14755=tf.reshape(var14754, [])
  #[]
  var14756=tf.add(var14752, var14755)
  #[512,21]
  var14757=tf.argmax(var14741, axis=2, output_type=tf.int32)
  #[512,21]
  var14758=tf.equal(var14757, var12351)
  #[512,21]
  var14759=tf.cast(var14758, tf.float32)
  #[512,21]
  var14760=tf.multiply(var14759, var14743)
  #[512,21]
  var14761=tf.reshape(var14760, [512,21])
  #[512]
  var14762=tf.reduce_sum(var14761, axis=1)
  #[512]
  var14763=tf.divide(var14762, var14748)
  #[512]
  var14764=tf.cast(var14763, tf.float32)
  #[512]
  var14765=tf.reshape(var14764, [512])
  #[]
  var14766=tf.reduce_mean(var14765, axis=0)
  #[10752,12]
  var14767=tf.reshape(var14741, [10752,12])
  #[10752,12]
  var14768=tf.nn.softmax(var14767, axis=1)
  #[512,21,12]
  var14769=tf.reshape(var14768, [512,21,12])
  return {"loss":var14756,"accuracy":var14766,"y_":var14769}
runModel = {"function":runModel_fn
           ,"batched":True
           ,"placeholders":{"x":{"shape":[512,21],"dtype":tf.int32}
                           ,"y":{"shape":[512,21],"dtype":tf.int32}
                           ,"weights":{"shape":[512,21],"dtype":tf.float32}}}