
import tensorflow as tf
def mkModel():
  
  #[12,90]
  var12345=tf.random.uniform([12,90], minval=-5.0e-2, maxval=5.0e-2, dtype=tf.float32) # 0
  var12346=tf.Variable(name="embs", trainable=True, initial_value=var12345)
  #[32,12]
  var12347=tf.random.uniform(
             [32,12], minval=-0.36927447, maxval=0.36927447, dtype=tf.float32) # 4
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
  #[512,32]
  var12353=tf.random.uniform([512,32], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[512,32]
  var12354=tf.floor(var12353)
  #[]
  var12355=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[32]
  var12356=tf.broadcast_to(tf.reshape(var12355, [1]), [32])
  #[32]
  var12357=tf.reshape(var12356, [32])
  #[512,32]
  var12358=tf.broadcast_to(tf.reshape(var12357, [1,32]), [512,32])
  #[512,32]
  var12359=tf.divide(var12354, var12358)
  #[]
  var12360=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[32]
  var12361=tf.broadcast_to(tf.reshape(var12360, [1]), [32])
  #[32]
  var12362=tf.reshape(var12361, [32])
  #[512,32]
  var12363=tf.broadcast_to(tf.reshape(var12362, [1,32]), [512,32])
  #[512,32]
  var12364=tf.cond(var12352, true_fn=lambda: var12359, false_fn=lambda: var12363)
  #[512,32]
  var12365=tf.random.uniform([512,32], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[512,32]
  var12366=tf.floor(var12365)
  #[512,32]
  var12367=tf.divide(var12366, var12358)
  #[512,32]
  var12368=tf.cond(var12352, true_fn=lambda: var12367, false_fn=lambda: var12363)
  #[]
  var12369=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var12370=tf.broadcast_to(tf.reshape(var12369, [1]), [1])
  #[]
  var12371=tf.reshape(var12370, [])
  #[32]
  var12372=tf.one_hot(var12371, axis=0, dtype=tf.float32, depth=32)
  #[512,32]
  var12373=tf.broadcast_to(tf.reshape(var12372, [1,32]), [512,32])
  #[512,32]
  var12374=tf.multiply(var12368, var12373)
  #[512,1,32]
  var12375=tf.reshape(var12374, [512,1,32])
  #[32]
  var12376=tf.range(start=0, limit=32, dtype=tf.int32)
  #[32,32]
  var12377=tf.broadcast_to(tf.reshape(var12376, [1,32]), [32,32])
  #[1024]
  var12378=tf.reshape(var12377, [1024])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var12379=tf.transpose(var12377, perm=[1,0])
  #[1024]
  var12380=tf.reshape(var12379, [1024])
  #[1024]
  var12381=tf.subtract(var12378, var12380)
  #[1024]
  var12382=tf.broadcast_to(tf.reshape(var12371, [1]), [1024])
  #[1024]
  var12383=tf.math.greater(var12381, var12382)
  #[]
  var12384=tf.constant(2, shape=[], dtype=tf.int32)
  #[1]
  var12385=tf.broadcast_to(tf.reshape(var12384, [1]), [1])
  #[]
  var12386=tf.reshape(var12385, [])
  #[]
  var12387=tf.constant(32, shape=[], dtype=tf.int32)
  #[1]
  var12388=tf.broadcast_to(tf.reshape(var12387, [1]), [1])
  #[]
  var12389=tf.reshape(var12388, [])
  #[]
  var12390=tf.multiply(var12386, var12389)
  #[1024]
  var12391=tf.broadcast_to(tf.reshape(var12390, [1]), [1024])
  #[1024]
  var12392=tf.subtract(var12391, var12380)
  #[]
  var12393=tf.constant(3, shape=[], dtype=tf.int32)
  #[1]
  var12394=tf.broadcast_to(tf.reshape(var12393, [1]), [1])
  #[]
  var12395=tf.reshape(var12394, [])
  #[1024]
  var12396=tf.broadcast_to(tf.reshape(var12395, [1]), [1024])
  #[1024]
  var12397=tf.subtract(var12392, var12396)
  #[1024]
  var12398=tf.multiply(var12380, var12397)
  #[1024]
  var12399=tf.broadcast_to(tf.reshape(var12386, [1]), [1024])
  #[1024]
  var12400=tf.math.floordiv(var12398, var12399)
  #[1024]
  var12401=tf.add(var12400, var12378)
  #[]
  var12402=tf.constant(1, shape=[], dtype=tf.int32)
  #[1]
  var12403=tf.broadcast_to(tf.reshape(var12402, [1]), [1])
  #[]
  var12404=tf.reshape(var12403, [])
  #[1024]
  var12405=tf.broadcast_to(tf.reshape(var12404, [1]), [1024])
  #[1024]
  var12406=tf.subtract(var12401, var12405)
  #[]
  var12407=tf.constant(90, shape=[], dtype=tf.int32)
  #[1]
  var12408=tf.broadcast_to(tf.reshape(var12407, [1]), [1])
  #[]
  var12409=tf.reshape(var12408, [])
  #[1024]
  var12410=tf.broadcast_to(tf.reshape(var12409, [1]), [1024])
  #[1024]
  var12411=tf.math.less(var12406, var12410)
  #[1024]
  var12412=tf.math.logical_and(var12383, var12411)
  #[512,1024]
  var12413=tf.broadcast_to(tf.reshape(var12412, [1,1024]), [512,1024])
  #[512,90]
  var12414=tf.random.uniform([512,90], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[512,90]
  var12415=tf.floor(var12414)
  #[90]
  var12416=tf.broadcast_to(tf.reshape(var12355, [1]), [90])
  #[90]
  var12417=tf.reshape(var12416, [90])
  #[512,90]
  var12418=tf.broadcast_to(tf.reshape(var12417, [1,90]), [512,90])
  #[512,90]
  var12419=tf.divide(var12415, var12418)
  #[90]
  var12420=tf.broadcast_to(tf.reshape(var12360, [1]), [90])
  #[90]
  var12421=tf.reshape(var12420, [90])
  #[512,90]
  var12422=tf.broadcast_to(tf.reshape(var12421, [1,90]), [512,90])
  #[512,90]
  var12423=tf.cond(var12352, true_fn=lambda: var12419, false_fn=lambda: var12422)
  #[12,90]
  var12424=embs
  #[512,21]
  var12425=x
  #[512,1]
  var12426=var12425[:,0:1]
  #[512]
  var12427=tf.reshape(var12426, [512])
  #[512,90]
  var12428=tf.gather(params=var12424, indices=var12427, batch_dims=0, axis=0)
  #[512,90]
  var12429=tf.multiply(var12423, var12428)
  #[512,1024]
  var12430=tf.broadcast_to(tf.reshape(var12406, [1,1024]), [512,1024])
  #[512,1024]
  var12431=tf.gather(params=var12429, indices=var12430, batch_dims=1, axis=1)
  #[]
  var12432=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var12433=tf.broadcast_to(tf.reshape(var12432, [1]), [1])
  #[]
  var12434=tf.reshape(var12433, [])
  #[1024]
  var12435=tf.broadcast_to(tf.reshape(var12434, [1]), [1024])
  #[512,1024]
  var12436=tf.broadcast_to(tf.reshape(var12435, [1,1024]), [512,1024])
  #[512,1024]
  var12437=tf.where(var12413, var12431, var12436)
  #[512,32,32]
  var12438=tf.reshape(var12437, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12439=tf.transpose(var12438, perm=[0,2,1])
  #[512,32,32]
  var12440=tf.subtract(var12438, var12439)
  #[512,32,32]
  var12441=tf.linalg.expm(var12440)
  #[512,1,32]
  var12442=tf.matmul(var12375, var12441)
  #[512,32]
  var12443=tf.reshape(var12442, [512,32])
  #[512,32]
  var12444=tf.multiply(var12364, var12443)
  #[512,32]
  var12445=tf.reshape(var12444, [512,32])
  #[32,12]
  var12446=projection_w
  #[512,12]
  var12447=tf.matmul(var12445, var12446)
  #[512,12]
  var12448=tf.reshape(var12447, [512,12])
  #[12]
  var12449=projection_bias
  #[512,12]
  var12450=tf.broadcast_to(tf.reshape(var12449, [1,12]), [512,12])
  #[512,12]
  var12451=tf.add(var12448, var12450)
  #[512,1,12]
  var12452=tf.reshape(var12451, [512,1,12])
  #[512,32]
  var12453=tf.multiply(var12368, var12443)
  #[512,1,32]
  var12454=tf.reshape(var12453, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12455=var12425[:,1:2]
  #[512]
  var12456=tf.reshape(var12455, [512])
  #[512,90]
  var12457=tf.gather(params=var12424, indices=var12456, batch_dims=0, axis=0)
  #[512,90]
  var12458=tf.multiply(var12423, var12457)
  #[512,1024]
  var12459=tf.gather(params=var12458, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12460=tf.where(var12413, var12459, var12436)
  #[512,32,32]
  var12461=tf.reshape(var12460, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12462=tf.transpose(var12461, perm=[0,2,1])
  #[512,32,32]
  var12463=tf.subtract(var12461, var12462)
  #[512,32,32]
  var12464=tf.linalg.expm(var12463)
  #[512,1,32]
  var12465=tf.matmul(var12454, var12464)
  #[512,32]
  var12466=tf.reshape(var12465, [512,32])
  #[512,32]
  var12467=tf.multiply(var12364, var12466)
  #[512,32]
  var12468=tf.reshape(var12467, [512,32])
  #[512,12]
  var12469=tf.matmul(var12468, var12446)
  #[512,12]
  var12470=tf.reshape(var12469, [512,12])
  #[512,12]
  var12471=tf.add(var12470, var12450)
  #[512,1,12]
  var12472=tf.reshape(var12471, [512,1,12])
  #[512,32]
  var12473=tf.multiply(var12368, var12466)
  #[512,1,32]
  var12474=tf.reshape(var12473, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12475=var12425[:,2:3]
  #[512]
  var12476=tf.reshape(var12475, [512])
  #[512,90]
  var12477=tf.gather(params=var12424, indices=var12476, batch_dims=0, axis=0)
  #[512,90]
  var12478=tf.multiply(var12423, var12477)
  #[512,1024]
  var12479=tf.gather(params=var12478, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12480=tf.where(var12413, var12479, var12436)
  #[512,32,32]
  var12481=tf.reshape(var12480, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12482=tf.transpose(var12481, perm=[0,2,1])
  #[512,32,32]
  var12483=tf.subtract(var12481, var12482)
  #[512,32,32]
  var12484=tf.linalg.expm(var12483)
  #[512,1,32]
  var12485=tf.matmul(var12474, var12484)
  #[512,32]
  var12486=tf.reshape(var12485, [512,32])
  #[512,32]
  var12487=tf.multiply(var12364, var12486)
  #[512,32]
  var12488=tf.reshape(var12487, [512,32])
  #[512,12]
  var12489=tf.matmul(var12488, var12446)
  #[512,12]
  var12490=tf.reshape(var12489, [512,12])
  #[512,12]
  var12491=tf.add(var12490, var12450)
  #[512,1,12]
  var12492=tf.reshape(var12491, [512,1,12])
  #[512,32]
  var12493=tf.multiply(var12368, var12486)
  #[512,1,32]
  var12494=tf.reshape(var12493, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12495=var12425[:,3:4]
  #[512]
  var12496=tf.reshape(var12495, [512])
  #[512,90]
  var12497=tf.gather(params=var12424, indices=var12496, batch_dims=0, axis=0)
  #[512,90]
  var12498=tf.multiply(var12423, var12497)
  #[512,1024]
  var12499=tf.gather(params=var12498, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12500=tf.where(var12413, var12499, var12436)
  #[512,32,32]
  var12501=tf.reshape(var12500, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12502=tf.transpose(var12501, perm=[0,2,1])
  #[512,32,32]
  var12503=tf.subtract(var12501, var12502)
  #[512,32,32]
  var12504=tf.linalg.expm(var12503)
  #[512,1,32]
  var12505=tf.matmul(var12494, var12504)
  #[512,32]
  var12506=tf.reshape(var12505, [512,32])
  #[512,32]
  var12507=tf.multiply(var12364, var12506)
  #[512,32]
  var12508=tf.reshape(var12507, [512,32])
  #[512,12]
  var12509=tf.matmul(var12508, var12446)
  #[512,12]
  var12510=tf.reshape(var12509, [512,12])
  #[512,12]
  var12511=tf.add(var12510, var12450)
  #[512,1,12]
  var12512=tf.reshape(var12511, [512,1,12])
  #[512,32]
  var12513=tf.multiply(var12368, var12506)
  #[512,1,32]
  var12514=tf.reshape(var12513, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12515=var12425[:,4:5]
  #[512]
  var12516=tf.reshape(var12515, [512])
  #[512,90]
  var12517=tf.gather(params=var12424, indices=var12516, batch_dims=0, axis=0)
  #[512,90]
  var12518=tf.multiply(var12423, var12517)
  #[512,1024]
  var12519=tf.gather(params=var12518, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12520=tf.where(var12413, var12519, var12436)
  #[512,32,32]
  var12521=tf.reshape(var12520, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12522=tf.transpose(var12521, perm=[0,2,1])
  #[512,32,32]
  var12523=tf.subtract(var12521, var12522)
  #[512,32,32]
  var12524=tf.linalg.expm(var12523)
  #[512,1,32]
  var12525=tf.matmul(var12514, var12524)
  #[512,32]
  var12526=tf.reshape(var12525, [512,32])
  #[512,32]
  var12527=tf.multiply(var12364, var12526)
  #[512,32]
  var12528=tf.reshape(var12527, [512,32])
  #[512,12]
  var12529=tf.matmul(var12528, var12446)
  #[512,12]
  var12530=tf.reshape(var12529, [512,12])
  #[512,12]
  var12531=tf.add(var12530, var12450)
  #[512,1,12]
  var12532=tf.reshape(var12531, [512,1,12])
  #[512,32]
  var12533=tf.multiply(var12368, var12526)
  #[512,1,32]
  var12534=tf.reshape(var12533, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12535=var12425[:,5:6]
  #[512]
  var12536=tf.reshape(var12535, [512])
  #[512,90]
  var12537=tf.gather(params=var12424, indices=var12536, batch_dims=0, axis=0)
  #[512,90]
  var12538=tf.multiply(var12423, var12537)
  #[512,1024]
  var12539=tf.gather(params=var12538, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12540=tf.where(var12413, var12539, var12436)
  #[512,32,32]
  var12541=tf.reshape(var12540, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12542=tf.transpose(var12541, perm=[0,2,1])
  #[512,32,32]
  var12543=tf.subtract(var12541, var12542)
  #[512,32,32]
  var12544=tf.linalg.expm(var12543)
  #[512,1,32]
  var12545=tf.matmul(var12534, var12544)
  #[512,32]
  var12546=tf.reshape(var12545, [512,32])
  #[512,32]
  var12547=tf.multiply(var12364, var12546)
  #[512,32]
  var12548=tf.reshape(var12547, [512,32])
  #[512,12]
  var12549=tf.matmul(var12548, var12446)
  #[512,12]
  var12550=tf.reshape(var12549, [512,12])
  #[512,12]
  var12551=tf.add(var12550, var12450)
  #[512,1,12]
  var12552=tf.reshape(var12551, [512,1,12])
  #[512,32]
  var12553=tf.multiply(var12368, var12546)
  #[512,1,32]
  var12554=tf.reshape(var12553, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12555=var12425[:,6:7]
  #[512]
  var12556=tf.reshape(var12555, [512])
  #[512,90]
  var12557=tf.gather(params=var12424, indices=var12556, batch_dims=0, axis=0)
  #[512,90]
  var12558=tf.multiply(var12423, var12557)
  #[512,1024]
  var12559=tf.gather(params=var12558, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12560=tf.where(var12413, var12559, var12436)
  #[512,32,32]
  var12561=tf.reshape(var12560, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12562=tf.transpose(var12561, perm=[0,2,1])
  #[512,32,32]
  var12563=tf.subtract(var12561, var12562)
  #[512,32,32]
  var12564=tf.linalg.expm(var12563)
  #[512,1,32]
  var12565=tf.matmul(var12554, var12564)
  #[512,32]
  var12566=tf.reshape(var12565, [512,32])
  #[512,32]
  var12567=tf.multiply(var12364, var12566)
  #[512,32]
  var12568=tf.reshape(var12567, [512,32])
  #[512,12]
  var12569=tf.matmul(var12568, var12446)
  #[512,12]
  var12570=tf.reshape(var12569, [512,12])
  #[512,12]
  var12571=tf.add(var12570, var12450)
  #[512,1,12]
  var12572=tf.reshape(var12571, [512,1,12])
  #[512,32]
  var12573=tf.multiply(var12368, var12566)
  #[512,1,32]
  var12574=tf.reshape(var12573, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12575=var12425[:,7:8]
  #[512]
  var12576=tf.reshape(var12575, [512])
  #[512,90]
  var12577=tf.gather(params=var12424, indices=var12576, batch_dims=0, axis=0)
  #[512,90]
  var12578=tf.multiply(var12423, var12577)
  #[512,1024]
  var12579=tf.gather(params=var12578, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12580=tf.where(var12413, var12579, var12436)
  #[512,32,32]
  var12581=tf.reshape(var12580, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12582=tf.transpose(var12581, perm=[0,2,1])
  #[512,32,32]
  var12583=tf.subtract(var12581, var12582)
  #[512,32,32]
  var12584=tf.linalg.expm(var12583)
  #[512,1,32]
  var12585=tf.matmul(var12574, var12584)
  #[512,32]
  var12586=tf.reshape(var12585, [512,32])
  #[512,32]
  var12587=tf.multiply(var12364, var12586)
  #[512,32]
  var12588=tf.reshape(var12587, [512,32])
  #[512,12]
  var12589=tf.matmul(var12588, var12446)
  #[512,12]
  var12590=tf.reshape(var12589, [512,12])
  #[512,12]
  var12591=tf.add(var12590, var12450)
  #[512,1,12]
  var12592=tf.reshape(var12591, [512,1,12])
  #[512,32]
  var12593=tf.multiply(var12368, var12586)
  #[512,1,32]
  var12594=tf.reshape(var12593, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12595=var12425[:,8:9]
  #[512]
  var12596=tf.reshape(var12595, [512])
  #[512,90]
  var12597=tf.gather(params=var12424, indices=var12596, batch_dims=0, axis=0)
  #[512,90]
  var12598=tf.multiply(var12423, var12597)
  #[512,1024]
  var12599=tf.gather(params=var12598, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12600=tf.where(var12413, var12599, var12436)
  #[512,32,32]
  var12601=tf.reshape(var12600, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12602=tf.transpose(var12601, perm=[0,2,1])
  #[512,32,32]
  var12603=tf.subtract(var12601, var12602)
  #[512,32,32]
  var12604=tf.linalg.expm(var12603)
  #[512,1,32]
  var12605=tf.matmul(var12594, var12604)
  #[512,32]
  var12606=tf.reshape(var12605, [512,32])
  #[512,32]
  var12607=tf.multiply(var12364, var12606)
  #[512,32]
  var12608=tf.reshape(var12607, [512,32])
  #[512,12]
  var12609=tf.matmul(var12608, var12446)
  #[512,12]
  var12610=tf.reshape(var12609, [512,12])
  #[512,12]
  var12611=tf.add(var12610, var12450)
  #[512,1,12]
  var12612=tf.reshape(var12611, [512,1,12])
  #[512,32]
  var12613=tf.multiply(var12368, var12606)
  #[512,1,32]
  var12614=tf.reshape(var12613, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12615=var12425[:,9:10]
  #[512]
  var12616=tf.reshape(var12615, [512])
  #[512,90]
  var12617=tf.gather(params=var12424, indices=var12616, batch_dims=0, axis=0)
  #[512,90]
  var12618=tf.multiply(var12423, var12617)
  #[512,1024]
  var12619=tf.gather(params=var12618, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12620=tf.where(var12413, var12619, var12436)
  #[512,32,32]
  var12621=tf.reshape(var12620, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12622=tf.transpose(var12621, perm=[0,2,1])
  #[512,32,32]
  var12623=tf.subtract(var12621, var12622)
  #[512,32,32]
  var12624=tf.linalg.expm(var12623)
  #[512,1,32]
  var12625=tf.matmul(var12614, var12624)
  #[512,32]
  var12626=tf.reshape(var12625, [512,32])
  #[512,32]
  var12627=tf.multiply(var12364, var12626)
  #[512,32]
  var12628=tf.reshape(var12627, [512,32])
  #[512,12]
  var12629=tf.matmul(var12628, var12446)
  #[512,12]
  var12630=tf.reshape(var12629, [512,12])
  #[512,12]
  var12631=tf.add(var12630, var12450)
  #[512,1,12]
  var12632=tf.reshape(var12631, [512,1,12])
  #[512,32]
  var12633=tf.multiply(var12368, var12626)
  #[512,1,32]
  var12634=tf.reshape(var12633, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12635=var12425[:,10:11]
  #[512]
  var12636=tf.reshape(var12635, [512])
  #[512,90]
  var12637=tf.gather(params=var12424, indices=var12636, batch_dims=0, axis=0)
  #[512,90]
  var12638=tf.multiply(var12423, var12637)
  #[512,1024]
  var12639=tf.gather(params=var12638, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12640=tf.where(var12413, var12639, var12436)
  #[512,32,32]
  var12641=tf.reshape(var12640, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12642=tf.transpose(var12641, perm=[0,2,1])
  #[512,32,32]
  var12643=tf.subtract(var12641, var12642)
  #[512,32,32]
  var12644=tf.linalg.expm(var12643)
  #[512,1,32]
  var12645=tf.matmul(var12634, var12644)
  #[512,32]
  var12646=tf.reshape(var12645, [512,32])
  #[512,32]
  var12647=tf.multiply(var12364, var12646)
  #[512,32]
  var12648=tf.reshape(var12647, [512,32])
  #[512,12]
  var12649=tf.matmul(var12648, var12446)
  #[512,12]
  var12650=tf.reshape(var12649, [512,12])
  #[512,12]
  var12651=tf.add(var12650, var12450)
  #[512,1,12]
  var12652=tf.reshape(var12651, [512,1,12])
  #[512,32]
  var12653=tf.multiply(var12368, var12646)
  #[512,1,32]
  var12654=tf.reshape(var12653, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12655=var12425[:,11:12]
  #[512]
  var12656=tf.reshape(var12655, [512])
  #[512,90]
  var12657=tf.gather(params=var12424, indices=var12656, batch_dims=0, axis=0)
  #[512,90]
  var12658=tf.multiply(var12423, var12657)
  #[512,1024]
  var12659=tf.gather(params=var12658, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12660=tf.where(var12413, var12659, var12436)
  #[512,32,32]
  var12661=tf.reshape(var12660, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12662=tf.transpose(var12661, perm=[0,2,1])
  #[512,32,32]
  var12663=tf.subtract(var12661, var12662)
  #[512,32,32]
  var12664=tf.linalg.expm(var12663)
  #[512,1,32]
  var12665=tf.matmul(var12654, var12664)
  #[512,32]
  var12666=tf.reshape(var12665, [512,32])
  #[512,32]
  var12667=tf.multiply(var12364, var12666)
  #[512,32]
  var12668=tf.reshape(var12667, [512,32])
  #[512,12]
  var12669=tf.matmul(var12668, var12446)
  #[512,12]
  var12670=tf.reshape(var12669, [512,12])
  #[512,12]
  var12671=tf.add(var12670, var12450)
  #[512,1,12]
  var12672=tf.reshape(var12671, [512,1,12])
  #[512,32]
  var12673=tf.multiply(var12368, var12666)
  #[512,1,32]
  var12674=tf.reshape(var12673, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12675=var12425[:,12:13]
  #[512]
  var12676=tf.reshape(var12675, [512])
  #[512,90]
  var12677=tf.gather(params=var12424, indices=var12676, batch_dims=0, axis=0)
  #[512,90]
  var12678=tf.multiply(var12423, var12677)
  #[512,1024]
  var12679=tf.gather(params=var12678, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12680=tf.where(var12413, var12679, var12436)
  #[512,32,32]
  var12681=tf.reshape(var12680, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12682=tf.transpose(var12681, perm=[0,2,1])
  #[512,32,32]
  var12683=tf.subtract(var12681, var12682)
  #[512,32,32]
  var12684=tf.linalg.expm(var12683)
  #[512,1,32]
  var12685=tf.matmul(var12674, var12684)
  #[512,32]
  var12686=tf.reshape(var12685, [512,32])
  #[512,32]
  var12687=tf.multiply(var12364, var12686)
  #[512,32]
  var12688=tf.reshape(var12687, [512,32])
  #[512,12]
  var12689=tf.matmul(var12688, var12446)
  #[512,12]
  var12690=tf.reshape(var12689, [512,12])
  #[512,12]
  var12691=tf.add(var12690, var12450)
  #[512,1,12]
  var12692=tf.reshape(var12691, [512,1,12])
  #[512,32]
  var12693=tf.multiply(var12368, var12686)
  #[512,1,32]
  var12694=tf.reshape(var12693, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12695=var12425[:,13:14]
  #[512]
  var12696=tf.reshape(var12695, [512])
  #[512,90]
  var12697=tf.gather(params=var12424, indices=var12696, batch_dims=0, axis=0)
  #[512,90]
  var12698=tf.multiply(var12423, var12697)
  #[512,1024]
  var12699=tf.gather(params=var12698, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12700=tf.where(var12413, var12699, var12436)
  #[512,32,32]
  var12701=tf.reshape(var12700, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12702=tf.transpose(var12701, perm=[0,2,1])
  #[512,32,32]
  var12703=tf.subtract(var12701, var12702)
  #[512,32,32]
  var12704=tf.linalg.expm(var12703)
  #[512,1,32]
  var12705=tf.matmul(var12694, var12704)
  #[512,32]
  var12706=tf.reshape(var12705, [512,32])
  #[512,32]
  var12707=tf.multiply(var12364, var12706)
  #[512,32]
  var12708=tf.reshape(var12707, [512,32])
  #[512,12]
  var12709=tf.matmul(var12708, var12446)
  #[512,12]
  var12710=tf.reshape(var12709, [512,12])
  #[512,12]
  var12711=tf.add(var12710, var12450)
  #[512,1,12]
  var12712=tf.reshape(var12711, [512,1,12])
  #[512,32]
  var12713=tf.multiply(var12368, var12706)
  #[512,1,32]
  var12714=tf.reshape(var12713, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12715=var12425[:,14:15]
  #[512]
  var12716=tf.reshape(var12715, [512])
  #[512,90]
  var12717=tf.gather(params=var12424, indices=var12716, batch_dims=0, axis=0)
  #[512,90]
  var12718=tf.multiply(var12423, var12717)
  #[512,1024]
  var12719=tf.gather(params=var12718, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12720=tf.where(var12413, var12719, var12436)
  #[512,32,32]
  var12721=tf.reshape(var12720, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12722=tf.transpose(var12721, perm=[0,2,1])
  #[512,32,32]
  var12723=tf.subtract(var12721, var12722)
  #[512,32,32]
  var12724=tf.linalg.expm(var12723)
  #[512,1,32]
  var12725=tf.matmul(var12714, var12724)
  #[512,32]
  var12726=tf.reshape(var12725, [512,32])
  #[512,32]
  var12727=tf.multiply(var12364, var12726)
  #[512,32]
  var12728=tf.reshape(var12727, [512,32])
  #[512,12]
  var12729=tf.matmul(var12728, var12446)
  #[512,12]
  var12730=tf.reshape(var12729, [512,12])
  #[512,12]
  var12731=tf.add(var12730, var12450)
  #[512,1,12]
  var12732=tf.reshape(var12731, [512,1,12])
  #[512,32]
  var12733=tf.multiply(var12368, var12726)
  #[512,1,32]
  var12734=tf.reshape(var12733, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12735=var12425[:,15:16]
  #[512]
  var12736=tf.reshape(var12735, [512])
  #[512,90]
  var12737=tf.gather(params=var12424, indices=var12736, batch_dims=0, axis=0)
  #[512,90]
  var12738=tf.multiply(var12423, var12737)
  #[512,1024]
  var12739=tf.gather(params=var12738, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12740=tf.where(var12413, var12739, var12436)
  #[512,32,32]
  var12741=tf.reshape(var12740, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12742=tf.transpose(var12741, perm=[0,2,1])
  #[512,32,32]
  var12743=tf.subtract(var12741, var12742)
  #[512,32,32]
  var12744=tf.linalg.expm(var12743)
  #[512,1,32]
  var12745=tf.matmul(var12734, var12744)
  #[512,32]
  var12746=tf.reshape(var12745, [512,32])
  #[512,32]
  var12747=tf.multiply(var12364, var12746)
  #[512,32]
  var12748=tf.reshape(var12747, [512,32])
  #[512,12]
  var12749=tf.matmul(var12748, var12446)
  #[512,12]
  var12750=tf.reshape(var12749, [512,12])
  #[512,12]
  var12751=tf.add(var12750, var12450)
  #[512,1,12]
  var12752=tf.reshape(var12751, [512,1,12])
  #[512,32]
  var12753=tf.multiply(var12368, var12746)
  #[512,1,32]
  var12754=tf.reshape(var12753, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12755=var12425[:,16:17]
  #[512]
  var12756=tf.reshape(var12755, [512])
  #[512,90]
  var12757=tf.gather(params=var12424, indices=var12756, batch_dims=0, axis=0)
  #[512,90]
  var12758=tf.multiply(var12423, var12757)
  #[512,1024]
  var12759=tf.gather(params=var12758, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12760=tf.where(var12413, var12759, var12436)
  #[512,32,32]
  var12761=tf.reshape(var12760, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12762=tf.transpose(var12761, perm=[0,2,1])
  #[512,32,32]
  var12763=tf.subtract(var12761, var12762)
  #[512,32,32]
  var12764=tf.linalg.expm(var12763)
  #[512,1,32]
  var12765=tf.matmul(var12754, var12764)
  #[512,32]
  var12766=tf.reshape(var12765, [512,32])
  #[512,32]
  var12767=tf.multiply(var12364, var12766)
  #[512,32]
  var12768=tf.reshape(var12767, [512,32])
  #[512,12]
  var12769=tf.matmul(var12768, var12446)
  #[512,12]
  var12770=tf.reshape(var12769, [512,12])
  #[512,12]
  var12771=tf.add(var12770, var12450)
  #[512,1,12]
  var12772=tf.reshape(var12771, [512,1,12])
  #[512,32]
  var12773=tf.multiply(var12368, var12766)
  #[512,1,32]
  var12774=tf.reshape(var12773, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12775=var12425[:,17:18]
  #[512]
  var12776=tf.reshape(var12775, [512])
  #[512,90]
  var12777=tf.gather(params=var12424, indices=var12776, batch_dims=0, axis=0)
  #[512,90]
  var12778=tf.multiply(var12423, var12777)
  #[512,1024]
  var12779=tf.gather(params=var12778, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12780=tf.where(var12413, var12779, var12436)
  #[512,32,32]
  var12781=tf.reshape(var12780, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12782=tf.transpose(var12781, perm=[0,2,1])
  #[512,32,32]
  var12783=tf.subtract(var12781, var12782)
  #[512,32,32]
  var12784=tf.linalg.expm(var12783)
  #[512,1,32]
  var12785=tf.matmul(var12774, var12784)
  #[512,32]
  var12786=tf.reshape(var12785, [512,32])
  #[512,32]
  var12787=tf.multiply(var12364, var12786)
  #[512,32]
  var12788=tf.reshape(var12787, [512,32])
  #[512,12]
  var12789=tf.matmul(var12788, var12446)
  #[512,12]
  var12790=tf.reshape(var12789, [512,12])
  #[512,12]
  var12791=tf.add(var12790, var12450)
  #[512,1,12]
  var12792=tf.reshape(var12791, [512,1,12])
  #[512,32]
  var12793=tf.multiply(var12368, var12786)
  #[512,1,32]
  var12794=tf.reshape(var12793, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12795=var12425[:,18:19]
  #[512]
  var12796=tf.reshape(var12795, [512])
  #[512,90]
  var12797=tf.gather(params=var12424, indices=var12796, batch_dims=0, axis=0)
  #[512,90]
  var12798=tf.multiply(var12423, var12797)
  #[512,1024]
  var12799=tf.gather(params=var12798, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12800=tf.where(var12413, var12799, var12436)
  #[512,32,32]
  var12801=tf.reshape(var12800, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12802=tf.transpose(var12801, perm=[0,2,1])
  #[512,32,32]
  var12803=tf.subtract(var12801, var12802)
  #[512,32,32]
  var12804=tf.linalg.expm(var12803)
  #[512,1,32]
  var12805=tf.matmul(var12794, var12804)
  #[512,32]
  var12806=tf.reshape(var12805, [512,32])
  #[512,32]
  var12807=tf.multiply(var12364, var12806)
  #[512,32]
  var12808=tf.reshape(var12807, [512,32])
  #[512,12]
  var12809=tf.matmul(var12808, var12446)
  #[512,12]
  var12810=tf.reshape(var12809, [512,12])
  #[512,12]
  var12811=tf.add(var12810, var12450)
  #[512,1,12]
  var12812=tf.reshape(var12811, [512,1,12])
  #[512,32]
  var12813=tf.multiply(var12368, var12806)
  #[512,1,32]
  var12814=tf.reshape(var12813, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12815=var12425[:,19:20]
  #[512]
  var12816=tf.reshape(var12815, [512])
  #[512,90]
  var12817=tf.gather(params=var12424, indices=var12816, batch_dims=0, axis=0)
  #[512,90]
  var12818=tf.multiply(var12423, var12817)
  #[512,1024]
  var12819=tf.gather(params=var12818, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12820=tf.where(var12413, var12819, var12436)
  #[512,32,32]
  var12821=tf.reshape(var12820, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12822=tf.transpose(var12821, perm=[0,2,1])
  #[512,32,32]
  var12823=tf.subtract(var12821, var12822)
  #[512,32,32]
  var12824=tf.linalg.expm(var12823)
  #[512,1,32]
  var12825=tf.matmul(var12814, var12824)
  #[512,32]
  var12826=tf.reshape(var12825, [512,32])
  #[512,32]
  var12827=tf.multiply(var12364, var12826)
  #[512,32]
  var12828=tf.reshape(var12827, [512,32])
  #[512,12]
  var12829=tf.matmul(var12828, var12446)
  #[512,12]
  var12830=tf.reshape(var12829, [512,12])
  #[512,12]
  var12831=tf.add(var12830, var12450)
  #[512,1,12]
  var12832=tf.reshape(var12831, [512,1,12])
  #[512,32]
  var12833=tf.multiply(var12368, var12826)
  #[512,1,32]
  var12834=tf.reshape(var12833, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12835=var12425[:,20:21]
  #[512]
  var12836=tf.reshape(var12835, [512])
  #[512,90]
  var12837=tf.gather(params=var12424, indices=var12836, batch_dims=0, axis=0)
  #[512,90]
  var12838=tf.multiply(var12423, var12837)
  #[512,1024]
  var12839=tf.gather(params=var12838, indices=var12430, batch_dims=1, axis=1)
  #[512,1024]
  var12840=tf.where(var12413, var12839, var12436)
  #[512,32,32]
  var12841=tf.reshape(var12840, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12842=tf.transpose(var12841, perm=[0,2,1])
  #[512,32,32]
  var12843=tf.subtract(var12841, var12842)
  #[512,32,32]
  var12844=tf.linalg.expm(var12843)
  #[512,1,32]
  var12845=tf.matmul(var12834, var12844)
  #[512,32]
  var12846=tf.reshape(var12845, [512,32])
  #[512,32]
  var12847=tf.multiply(var12364, var12846)
  #[512,32]
  var12848=tf.reshape(var12847, [512,32])
  #[512,12]
  var12849=tf.matmul(var12848, var12446)
  #[512,12]
  var12850=tf.reshape(var12849, [512,12])
  #[512,12]
  var12851=tf.add(var12850, var12450)
  #[512,1,12]
  var12852=tf.reshape(var12851, [512,1,12])
  #[512,21,12]
  var12853=tf.concat([var12452
                     ,var12472
                     ,var12492
                     ,var12512
                     ,var12532
                     ,var12552
                     ,var12572
                     ,var12592
                     ,var12612
                     ,var12632
                     ,var12652
                     ,var12672
                     ,var12692
                     ,var12712
                     ,var12732
                     ,var12752
                     ,var12772
                     ,var12792
                     ,var12812
                     ,var12832
                     ,var12852],
                     axis=1)
  #[512,21]
  var12854=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=var12351, logits=var12853)
  #[512,21]
  var12855=weights
  #[512,21]
  var12856=tf.multiply(var12854, var12855)
  #[512,21]
  var12857=tf.reshape(var12856, [512,21])
  #[512]
  var12858=tf.reduce_sum(var12857, axis=1)
  #[512,21]
  var12859=tf.reshape(var12855, [512,21])
  #[512]
  var12860=tf.reduce_sum(var12859, axis=1)
  #[512]
  var12861=tf.divide(var12858, var12860)
  #[512]
  var12862=tf.cast(var12861, tf.float32)
  #[512]
  var12863=tf.reshape(var12862, [512])
  #[]
  var12864=tf.reduce_mean(var12863, axis=0)
  #[]
  var12865=tf.add(var12864, var12434)
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,21]
  var12866=tf.argmax(var12853, axis=2, output_type=tf.int32)
  #[512,21]
  var12867=tf.equal(var12866, var12351)
  #[512,21]
  var12868=tf.cast(var12867, tf.float32)
  #[512,21]
  var12869=tf.multiply(var12868, var12855)
  #[512,21]
  var12870=tf.reshape(var12869, [512,21])
  #[512]
  var12871=tf.reduce_sum(var12870, axis=1)
  #[512]
  var12872=tf.divide(var12871, var12860)
  #[512]
  var12873=tf.cast(var12872, tf.float32)
  #[512]
  var12874=tf.reshape(var12873, [512])
  #[]
  var12875=tf.reduce_mean(var12874, axis=0)
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[10752,12]
  var12876=tf.reshape(var12853, [10752,12])
  #[10752,12]
  var12877=tf.nn.softmax(var12876, axis=1)
  #[512,21,12]
  var12878=tf.reshape(var12877, [512,21,12])
  return {"loss":var12865,"accuracy":var12875,"y_":var12878}
runModel = {"function":runModel_fn
           ,"batched":True
           ,"placeholders":{"x":{"shape":[512,21],"dtype":tf.int32}
                           ,"y":{"shape":[512,21],"dtype":tf.int32}
                           ,"weights":{"shape":[512,21],"dtype":tf.float32}}}
@tf.function
def probeStates_fn(training_placeholder, embs, projection_w, projection_bias, x):
  
  #[]
  var12879=training_placeholder
  #[32]
  var12880=tf.random.uniform([32], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[32]
  var12881=tf.floor(var12880)
  #[]
  var12882=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[32]
  var12883=tf.broadcast_to(tf.reshape(var12882, [1]), [32])
  #[32]
  var12884=tf.reshape(var12883, [32])
  #[32]
  var12885=tf.divide(var12881, var12884)
  #[]
  var12886=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[32]
  var12887=tf.broadcast_to(tf.reshape(var12886, [1]), [32])
  #[32]
  var12888=tf.reshape(var12887, [32])
  #[32]
  var12889=tf.cond(var12879, true_fn=lambda: var12885, false_fn=lambda: var12888)
  #[32]
  var12890=tf.random.uniform([32], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[32]
  var12891=tf.floor(var12890)
  #[32]
  var12892=tf.divide(var12891, var12884)
  #[32]
  var12893=tf.cond(var12879, true_fn=lambda: var12892, false_fn=lambda: var12888)
  #[]
  var12894=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var12895=tf.broadcast_to(tf.reshape(var12894, [1]), [1])
  #[]
  var12896=tf.reshape(var12895, [])
  #[32]
  var12897=tf.one_hot(var12896, axis=0, dtype=tf.float32, depth=32)
  #[32]
  var12898=tf.multiply(var12893, var12897)
  #[1,32]
  var12899=tf.reshape(var12898, [1,32])
  #[32]
  var12900=tf.range(start=0, limit=32, dtype=tf.int32)
  #[32,32]
  var12901=tf.broadcast_to(tf.reshape(var12900, [1,32]), [32,32])
  #[1024]
  var12902=tf.reshape(var12901, [1024])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var12903=tf.transpose(var12901, perm=[1,0])
  #[1024]
  var12904=tf.reshape(var12903, [1024])
  #[1024]
  var12905=tf.subtract(var12902, var12904)
  #[1024]
  var12906=tf.broadcast_to(tf.reshape(var12896, [1]), [1024])
  #[1024]
  var12907=tf.math.greater(var12905, var12906)
  #transpose: p = PermSwap; [32,32]
  #[]
  var12908=tf.constant(2, shape=[], dtype=tf.int32)
  #[1]
  var12909=tf.broadcast_to(tf.reshape(var12908, [1]), [1])
  #[]
  var12910=tf.reshape(var12909, [])
  #[]
  var12911=tf.constant(32, shape=[], dtype=tf.int32)
  #[1]
  var12912=tf.broadcast_to(tf.reshape(var12911, [1]), [1])
  #[]
  var12913=tf.reshape(var12912, [])
  #[]
  var12914=tf.multiply(var12910, var12913)
  #[1024]
  var12915=tf.broadcast_to(tf.reshape(var12914, [1]), [1024])
  #[1024]
  var12916=tf.subtract(var12915, var12904)
  #[]
  var12917=tf.constant(3, shape=[], dtype=tf.int32)
  #[1]
  var12918=tf.broadcast_to(tf.reshape(var12917, [1]), [1])
  #[]
  var12919=tf.reshape(var12918, [])
  #[1024]
  var12920=tf.broadcast_to(tf.reshape(var12919, [1]), [1024])
  #[1024]
  var12921=tf.subtract(var12916, var12920)
  #[1024]
  var12922=tf.multiply(var12904, var12921)
  #[1024]
  var12923=tf.broadcast_to(tf.reshape(var12910, [1]), [1024])
  #[1024]
  var12924=tf.math.floordiv(var12922, var12923)
  #[1024]
  var12925=tf.add(var12924, var12902)
  #[]
  var12926=tf.constant(1, shape=[], dtype=tf.int32)
  #[1]
  var12927=tf.broadcast_to(tf.reshape(var12926, [1]), [1])
  #[]
  var12928=tf.reshape(var12927, [])
  #[1024]
  var12929=tf.broadcast_to(tf.reshape(var12928, [1]), [1024])
  #[1024]
  var12930=tf.subtract(var12925, var12929)
  #[]
  var12931=tf.constant(90, shape=[], dtype=tf.int32)
  #[1]
  var12932=tf.broadcast_to(tf.reshape(var12931, [1]), [1])
  #[]
  var12933=tf.reshape(var12932, [])
  #[1024]
  var12934=tf.broadcast_to(tf.reshape(var12933, [1]), [1024])
  #[1024]
  var12935=tf.math.less(var12930, var12934)
  #[1024]
  var12936=tf.math.logical_and(var12907, var12935)
  #[90]
  var12937=tf.random.uniform([90], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[90]
  var12938=tf.floor(var12937)
  #[90]
  var12939=tf.broadcast_to(tf.reshape(var12882, [1]), [90])
  #[90]
  var12940=tf.reshape(var12939, [90])
  #[90]
  var12941=tf.divide(var12938, var12940)
  #[90]
  var12942=tf.broadcast_to(tf.reshape(var12886, [1]), [90])
  #[90]
  var12943=tf.reshape(var12942, [90])
  #[90]
  var12944=tf.cond(var12879, true_fn=lambda: var12941, false_fn=lambda: var12943)
  #[12,90]
  var12945=embs
  #[21]
  var12946=x
  #[1]
  var12947=var12946[0:1]
  #[]
  var12948=tf.reshape(var12947, [])
  #[90]
  var12949=tf.gather(params=var12945, indices=var12948, batch_dims=0, axis=0)
  #[90]
  var12950=tf.multiply(var12944, var12949)
  #[1024]
  var12951=tf.gather(params=var12950, indices=var12930, batch_dims=0, axis=0)
  #[]
  var12952=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var12953=tf.broadcast_to(tf.reshape(var12952, [1]), [1])
  #[]
  var12954=tf.reshape(var12953, [])
  #[1024]
  var12955=tf.broadcast_to(tf.reshape(var12954, [1]), [1024])
  #[1024]
  var12956=tf.where(var12936, var12951, var12955)
  #[32,32]
  var12957=tf.reshape(var12956, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var12958=tf.transpose(var12957, perm=[1,0])
  #[32,32]
  var12959=tf.subtract(var12957, var12958)
  #[32,32]
  var12960=tf.linalg.expm(var12959)
  #[1,32]
  var12961=tf.matmul(var12899, var12960)
  #[32]
  var12962=tf.reshape(var12961, [32])
  #[32]
  var12963=tf.multiply(var12889, var12962)
  #[1,32]
  var12964=tf.reshape(var12963, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var12965=tf.multiply(var12893, var12962)
  #[1,32]
  var12966=tf.reshape(var12965, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var12967=var12946[1:2]
  #[]
  var12968=tf.reshape(var12967, [])
  #[90]
  var12969=tf.gather(params=var12945, indices=var12968, batch_dims=0, axis=0)
  #[90]
  var12970=tf.multiply(var12944, var12969)
  #[1024]
  var12971=tf.gather(params=var12970, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var12972=tf.where(var12936, var12971, var12955)
  #[32,32]
  var12973=tf.reshape(var12972, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var12974=tf.transpose(var12973, perm=[1,0])
  #[32,32]
  var12975=tf.subtract(var12973, var12974)
  #[32,32]
  var12976=tf.linalg.expm(var12975)
  #[1,32]
  var12977=tf.matmul(var12966, var12976)
  #[32]
  var12978=tf.reshape(var12977, [32])
  #[32]
  var12979=tf.multiply(var12889, var12978)
  #[1,32]
  var12980=tf.reshape(var12979, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var12981=tf.multiply(var12893, var12978)
  #[1,32]
  var12982=tf.reshape(var12981, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var12983=var12946[2:3]
  #[]
  var12984=tf.reshape(var12983, [])
  #[90]
  var12985=tf.gather(params=var12945, indices=var12984, batch_dims=0, axis=0)
  #[90]
  var12986=tf.multiply(var12944, var12985)
  #[1024]
  var12987=tf.gather(params=var12986, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var12988=tf.where(var12936, var12987, var12955)
  #[32,32]
  var12989=tf.reshape(var12988, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var12990=tf.transpose(var12989, perm=[1,0])
  #[32,32]
  var12991=tf.subtract(var12989, var12990)
  #[32,32]
  var12992=tf.linalg.expm(var12991)
  #[1,32]
  var12993=tf.matmul(var12982, var12992)
  #[32]
  var12994=tf.reshape(var12993, [32])
  #[32]
  var12995=tf.multiply(var12889, var12994)
  #[1,32]
  var12996=tf.reshape(var12995, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var12997=tf.multiply(var12893, var12994)
  #[1,32]
  var12998=tf.reshape(var12997, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var12999=var12946[3:4]
  #[]
  var13000=tf.reshape(var12999, [])
  #[90]
  var13001=tf.gather(params=var12945, indices=var13000, batch_dims=0, axis=0)
  #[90]
  var13002=tf.multiply(var12944, var13001)
  #[1024]
  var13003=tf.gather(params=var13002, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13004=tf.where(var12936, var13003, var12955)
  #[32,32]
  var13005=tf.reshape(var13004, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13006=tf.transpose(var13005, perm=[1,0])
  #[32,32]
  var13007=tf.subtract(var13005, var13006)
  #[32,32]
  var13008=tf.linalg.expm(var13007)
  #[1,32]
  var13009=tf.matmul(var12998, var13008)
  #[32]
  var13010=tf.reshape(var13009, [32])
  #[32]
  var13011=tf.multiply(var12889, var13010)
  #[1,32]
  var13012=tf.reshape(var13011, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13013=tf.multiply(var12893, var13010)
  #[1,32]
  var13014=tf.reshape(var13013, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13015=var12946[4:5]
  #[]
  var13016=tf.reshape(var13015, [])
  #[90]
  var13017=tf.gather(params=var12945, indices=var13016, batch_dims=0, axis=0)
  #[90]
  var13018=tf.multiply(var12944, var13017)
  #[1024]
  var13019=tf.gather(params=var13018, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13020=tf.where(var12936, var13019, var12955)
  #[32,32]
  var13021=tf.reshape(var13020, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13022=tf.transpose(var13021, perm=[1,0])
  #[32,32]
  var13023=tf.subtract(var13021, var13022)
  #[32,32]
  var13024=tf.linalg.expm(var13023)
  #[1,32]
  var13025=tf.matmul(var13014, var13024)
  #[32]
  var13026=tf.reshape(var13025, [32])
  #[32]
  var13027=tf.multiply(var12889, var13026)
  #[1,32]
  var13028=tf.reshape(var13027, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13029=tf.multiply(var12893, var13026)
  #[1,32]
  var13030=tf.reshape(var13029, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13031=var12946[5:6]
  #[]
  var13032=tf.reshape(var13031, [])
  #[90]
  var13033=tf.gather(params=var12945, indices=var13032, batch_dims=0, axis=0)
  #[90]
  var13034=tf.multiply(var12944, var13033)
  #[1024]
  var13035=tf.gather(params=var13034, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13036=tf.where(var12936, var13035, var12955)
  #[32,32]
  var13037=tf.reshape(var13036, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13038=tf.transpose(var13037, perm=[1,0])
  #[32,32]
  var13039=tf.subtract(var13037, var13038)
  #[32,32]
  var13040=tf.linalg.expm(var13039)
  #[1,32]
  var13041=tf.matmul(var13030, var13040)
  #[32]
  var13042=tf.reshape(var13041, [32])
  #[32]
  var13043=tf.multiply(var12889, var13042)
  #[1,32]
  var13044=tf.reshape(var13043, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13045=tf.multiply(var12893, var13042)
  #[1,32]
  var13046=tf.reshape(var13045, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13047=var12946[6:7]
  #[]
  var13048=tf.reshape(var13047, [])
  #[90]
  var13049=tf.gather(params=var12945, indices=var13048, batch_dims=0, axis=0)
  #[90]
  var13050=tf.multiply(var12944, var13049)
  #[1024]
  var13051=tf.gather(params=var13050, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13052=tf.where(var12936, var13051, var12955)
  #[32,32]
  var13053=tf.reshape(var13052, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13054=tf.transpose(var13053, perm=[1,0])
  #[32,32]
  var13055=tf.subtract(var13053, var13054)
  #[32,32]
  var13056=tf.linalg.expm(var13055)
  #[1,32]
  var13057=tf.matmul(var13046, var13056)
  #[32]
  var13058=tf.reshape(var13057, [32])
  #[32]
  var13059=tf.multiply(var12889, var13058)
  #[1,32]
  var13060=tf.reshape(var13059, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13061=tf.multiply(var12893, var13058)
  #[1,32]
  var13062=tf.reshape(var13061, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13063=var12946[7:8]
  #[]
  var13064=tf.reshape(var13063, [])
  #[90]
  var13065=tf.gather(params=var12945, indices=var13064, batch_dims=0, axis=0)
  #[90]
  var13066=tf.multiply(var12944, var13065)
  #[1024]
  var13067=tf.gather(params=var13066, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13068=tf.where(var12936, var13067, var12955)
  #[32,32]
  var13069=tf.reshape(var13068, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13070=tf.transpose(var13069, perm=[1,0])
  #[32,32]
  var13071=tf.subtract(var13069, var13070)
  #[32,32]
  var13072=tf.linalg.expm(var13071)
  #[1,32]
  var13073=tf.matmul(var13062, var13072)
  #[32]
  var13074=tf.reshape(var13073, [32])
  #[32]
  var13075=tf.multiply(var12889, var13074)
  #[1,32]
  var13076=tf.reshape(var13075, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13077=tf.multiply(var12893, var13074)
  #[1,32]
  var13078=tf.reshape(var13077, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13079=var12946[8:9]
  #[]
  var13080=tf.reshape(var13079, [])
  #[90]
  var13081=tf.gather(params=var12945, indices=var13080, batch_dims=0, axis=0)
  #[90]
  var13082=tf.multiply(var12944, var13081)
  #[1024]
  var13083=tf.gather(params=var13082, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13084=tf.where(var12936, var13083, var12955)
  #[32,32]
  var13085=tf.reshape(var13084, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13086=tf.transpose(var13085, perm=[1,0])
  #[32,32]
  var13087=tf.subtract(var13085, var13086)
  #[32,32]
  var13088=tf.linalg.expm(var13087)
  #[1,32]
  var13089=tf.matmul(var13078, var13088)
  #[32]
  var13090=tf.reshape(var13089, [32])
  #[32]
  var13091=tf.multiply(var12889, var13090)
  #[1,32]
  var13092=tf.reshape(var13091, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13093=tf.multiply(var12893, var13090)
  #[1,32]
  var13094=tf.reshape(var13093, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13095=var12946[9:10]
  #[]
  var13096=tf.reshape(var13095, [])
  #[90]
  var13097=tf.gather(params=var12945, indices=var13096, batch_dims=0, axis=0)
  #[90]
  var13098=tf.multiply(var12944, var13097)
  #[1024]
  var13099=tf.gather(params=var13098, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13100=tf.where(var12936, var13099, var12955)
  #[32,32]
  var13101=tf.reshape(var13100, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13102=tf.transpose(var13101, perm=[1,0])
  #[32,32]
  var13103=tf.subtract(var13101, var13102)
  #[32,32]
  var13104=tf.linalg.expm(var13103)
  #[1,32]
  var13105=tf.matmul(var13094, var13104)
  #[32]
  var13106=tf.reshape(var13105, [32])
  #[32]
  var13107=tf.multiply(var12889, var13106)
  #[1,32]
  var13108=tf.reshape(var13107, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13109=tf.multiply(var12893, var13106)
  #[1,32]
  var13110=tf.reshape(var13109, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13111=var12946[10:11]
  #[]
  var13112=tf.reshape(var13111, [])
  #[90]
  var13113=tf.gather(params=var12945, indices=var13112, batch_dims=0, axis=0)
  #[90]
  var13114=tf.multiply(var12944, var13113)
  #[1024]
  var13115=tf.gather(params=var13114, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13116=tf.where(var12936, var13115, var12955)
  #[32,32]
  var13117=tf.reshape(var13116, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13118=tf.transpose(var13117, perm=[1,0])
  #[32,32]
  var13119=tf.subtract(var13117, var13118)
  #[32,32]
  var13120=tf.linalg.expm(var13119)
  #[1,32]
  var13121=tf.matmul(var13110, var13120)
  #[32]
  var13122=tf.reshape(var13121, [32])
  #[32]
  var13123=tf.multiply(var12889, var13122)
  #[1,32]
  var13124=tf.reshape(var13123, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13125=tf.multiply(var12893, var13122)
  #[1,32]
  var13126=tf.reshape(var13125, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13127=var12946[11:12]
  #[]
  var13128=tf.reshape(var13127, [])
  #[90]
  var13129=tf.gather(params=var12945, indices=var13128, batch_dims=0, axis=0)
  #[90]
  var13130=tf.multiply(var12944, var13129)
  #[1024]
  var13131=tf.gather(params=var13130, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13132=tf.where(var12936, var13131, var12955)
  #[32,32]
  var13133=tf.reshape(var13132, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13134=tf.transpose(var13133, perm=[1,0])
  #[32,32]
  var13135=tf.subtract(var13133, var13134)
  #[32,32]
  var13136=tf.linalg.expm(var13135)
  #[1,32]
  var13137=tf.matmul(var13126, var13136)
  #[32]
  var13138=tf.reshape(var13137, [32])
  #[32]
  var13139=tf.multiply(var12889, var13138)
  #[1,32]
  var13140=tf.reshape(var13139, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13141=tf.multiply(var12893, var13138)
  #[1,32]
  var13142=tf.reshape(var13141, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13143=var12946[12:13]
  #[]
  var13144=tf.reshape(var13143, [])
  #[90]
  var13145=tf.gather(params=var12945, indices=var13144, batch_dims=0, axis=0)
  #[90]
  var13146=tf.multiply(var12944, var13145)
  #[1024]
  var13147=tf.gather(params=var13146, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13148=tf.where(var12936, var13147, var12955)
  #[32,32]
  var13149=tf.reshape(var13148, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13150=tf.transpose(var13149, perm=[1,0])
  #[32,32]
  var13151=tf.subtract(var13149, var13150)
  #[32,32]
  var13152=tf.linalg.expm(var13151)
  #[1,32]
  var13153=tf.matmul(var13142, var13152)
  #[32]
  var13154=tf.reshape(var13153, [32])
  #[32]
  var13155=tf.multiply(var12889, var13154)
  #[1,32]
  var13156=tf.reshape(var13155, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13157=tf.multiply(var12893, var13154)
  #[1,32]
  var13158=tf.reshape(var13157, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13159=var12946[13:14]
  #[]
  var13160=tf.reshape(var13159, [])
  #[90]
  var13161=tf.gather(params=var12945, indices=var13160, batch_dims=0, axis=0)
  #[90]
  var13162=tf.multiply(var12944, var13161)
  #[1024]
  var13163=tf.gather(params=var13162, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13164=tf.where(var12936, var13163, var12955)
  #[32,32]
  var13165=tf.reshape(var13164, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13166=tf.transpose(var13165, perm=[1,0])
  #[32,32]
  var13167=tf.subtract(var13165, var13166)
  #[32,32]
  var13168=tf.linalg.expm(var13167)
  #[1,32]
  var13169=tf.matmul(var13158, var13168)
  #[32]
  var13170=tf.reshape(var13169, [32])
  #[32]
  var13171=tf.multiply(var12889, var13170)
  #[1,32]
  var13172=tf.reshape(var13171, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13173=tf.multiply(var12893, var13170)
  #[1,32]
  var13174=tf.reshape(var13173, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13175=var12946[14:15]
  #[]
  var13176=tf.reshape(var13175, [])
  #[90]
  var13177=tf.gather(params=var12945, indices=var13176, batch_dims=0, axis=0)
  #[90]
  var13178=tf.multiply(var12944, var13177)
  #[1024]
  var13179=tf.gather(params=var13178, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13180=tf.where(var12936, var13179, var12955)
  #[32,32]
  var13181=tf.reshape(var13180, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13182=tf.transpose(var13181, perm=[1,0])
  #[32,32]
  var13183=tf.subtract(var13181, var13182)
  #[32,32]
  var13184=tf.linalg.expm(var13183)
  #[1,32]
  var13185=tf.matmul(var13174, var13184)
  #[32]
  var13186=tf.reshape(var13185, [32])
  #[32]
  var13187=tf.multiply(var12889, var13186)
  #[1,32]
  var13188=tf.reshape(var13187, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13189=tf.multiply(var12893, var13186)
  #[1,32]
  var13190=tf.reshape(var13189, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13191=var12946[15:16]
  #[]
  var13192=tf.reshape(var13191, [])
  #[90]
  var13193=tf.gather(params=var12945, indices=var13192, batch_dims=0, axis=0)
  #[90]
  var13194=tf.multiply(var12944, var13193)
  #[1024]
  var13195=tf.gather(params=var13194, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13196=tf.where(var12936, var13195, var12955)
  #[32,32]
  var13197=tf.reshape(var13196, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13198=tf.transpose(var13197, perm=[1,0])
  #[32,32]
  var13199=tf.subtract(var13197, var13198)
  #[32,32]
  var13200=tf.linalg.expm(var13199)
  #[1,32]
  var13201=tf.matmul(var13190, var13200)
  #[32]
  var13202=tf.reshape(var13201, [32])
  #[32]
  var13203=tf.multiply(var12889, var13202)
  #[1,32]
  var13204=tf.reshape(var13203, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13205=tf.multiply(var12893, var13202)
  #[1,32]
  var13206=tf.reshape(var13205, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13207=var12946[16:17]
  #[]
  var13208=tf.reshape(var13207, [])
  #[90]
  var13209=tf.gather(params=var12945, indices=var13208, batch_dims=0, axis=0)
  #[90]
  var13210=tf.multiply(var12944, var13209)
  #[1024]
  var13211=tf.gather(params=var13210, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13212=tf.where(var12936, var13211, var12955)
  #[32,32]
  var13213=tf.reshape(var13212, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13214=tf.transpose(var13213, perm=[1,0])
  #[32,32]
  var13215=tf.subtract(var13213, var13214)
  #[32,32]
  var13216=tf.linalg.expm(var13215)
  #[1,32]
  var13217=tf.matmul(var13206, var13216)
  #[32]
  var13218=tf.reshape(var13217, [32])
  #[32]
  var13219=tf.multiply(var12889, var13218)
  #[1,32]
  var13220=tf.reshape(var13219, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13221=tf.multiply(var12893, var13218)
  #[1,32]
  var13222=tf.reshape(var13221, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13223=var12946[17:18]
  #[]
  var13224=tf.reshape(var13223, [])
  #[90]
  var13225=tf.gather(params=var12945, indices=var13224, batch_dims=0, axis=0)
  #[90]
  var13226=tf.multiply(var12944, var13225)
  #[1024]
  var13227=tf.gather(params=var13226, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13228=tf.where(var12936, var13227, var12955)
  #[32,32]
  var13229=tf.reshape(var13228, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13230=tf.transpose(var13229, perm=[1,0])
  #[32,32]
  var13231=tf.subtract(var13229, var13230)
  #[32,32]
  var13232=tf.linalg.expm(var13231)
  #[1,32]
  var13233=tf.matmul(var13222, var13232)
  #[32]
  var13234=tf.reshape(var13233, [32])
  #[32]
  var13235=tf.multiply(var12889, var13234)
  #[1,32]
  var13236=tf.reshape(var13235, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13237=tf.multiply(var12893, var13234)
  #[1,32]
  var13238=tf.reshape(var13237, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13239=var12946[18:19]
  #[]
  var13240=tf.reshape(var13239, [])
  #[90]
  var13241=tf.gather(params=var12945, indices=var13240, batch_dims=0, axis=0)
  #[90]
  var13242=tf.multiply(var12944, var13241)
  #[1024]
  var13243=tf.gather(params=var13242, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13244=tf.where(var12936, var13243, var12955)
  #[32,32]
  var13245=tf.reshape(var13244, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13246=tf.transpose(var13245, perm=[1,0])
  #[32,32]
  var13247=tf.subtract(var13245, var13246)
  #[32,32]
  var13248=tf.linalg.expm(var13247)
  #[1,32]
  var13249=tf.matmul(var13238, var13248)
  #[32]
  var13250=tf.reshape(var13249, [32])
  #[32]
  var13251=tf.multiply(var12889, var13250)
  #[1,32]
  var13252=tf.reshape(var13251, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13253=tf.multiply(var12893, var13250)
  #[1,32]
  var13254=tf.reshape(var13253, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13255=var12946[19:20]
  #[]
  var13256=tf.reshape(var13255, [])
  #[90]
  var13257=tf.gather(params=var12945, indices=var13256, batch_dims=0, axis=0)
  #[90]
  var13258=tf.multiply(var12944, var13257)
  #[1024]
  var13259=tf.gather(params=var13258, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13260=tf.where(var12936, var13259, var12955)
  #[32,32]
  var13261=tf.reshape(var13260, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13262=tf.transpose(var13261, perm=[1,0])
  #[32,32]
  var13263=tf.subtract(var13261, var13262)
  #[32,32]
  var13264=tf.linalg.expm(var13263)
  #[1,32]
  var13265=tf.matmul(var13254, var13264)
  #[32]
  var13266=tf.reshape(var13265, [32])
  #[32]
  var13267=tf.multiply(var12889, var13266)
  #[1,32]
  var13268=tf.reshape(var13267, [1,32])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13269=tf.multiply(var12893, var13266)
  #[1,32]
  var13270=tf.reshape(var13269, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13271=var12946[20:21]
  #[]
  var13272=tf.reshape(var13271, [])
  #[90]
  var13273=tf.gather(params=var12945, indices=var13272, batch_dims=0, axis=0)
  #[90]
  var13274=tf.multiply(var12944, var13273)
  #[1024]
  var13275=tf.gather(params=var13274, indices=var12930, batch_dims=0, axis=0)
  #[1024]
  var13276=tf.where(var12936, var13275, var12955)
  #[32,32]
  var13277=tf.reshape(var13276, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13278=tf.transpose(var13277, perm=[1,0])
  #[32,32]
  var13279=tf.subtract(var13277, var13278)
  #[32,32]
  var13280=tf.linalg.expm(var13279)
  #[1,32]
  var13281=tf.matmul(var13270, var13280)
  #[32]
  var13282=tf.reshape(var13281, [32])
  #[32]
  var13283=tf.multiply(var12889, var13282)
  #[1,32]
  var13284=tf.reshape(var13283, [1,32])
  #[21,32]
  var13285=tf.concat([var12964
                     ,var12980
                     ,var12996
                     ,var13012
                     ,var13028
                     ,var13044
                     ,var13060
                     ,var13076
                     ,var13092
                     ,var13108
                     ,var13124
                     ,var13140
                     ,var13156
                     ,var13172
                     ,var13188
                     ,var13204
                     ,var13220
                     ,var13236
                     ,var13252
                     ,var13268
                     ,var13284],
                     axis=0)
  return {"states":var13285}
probeStates = {"function":probeStates_fn
              ,"batched":False
              ,"placeholders":{"x":{"shape":[21],"dtype":tf.int32}}}
@tf.function
def probePreds_fn(training_placeholder, embs, projection_w, projection_bias, x):
  
  #[]
  var13286=training_placeholder
  #[32]
  var13287=tf.random.uniform([32], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[32]
  var13288=tf.floor(var13287)
  #[]
  var13289=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[32]
  var13290=tf.broadcast_to(tf.reshape(var13289, [1]), [32])
  #[32]
  var13291=tf.reshape(var13290, [32])
  #[32]
  var13292=tf.divide(var13288, var13291)
  #[]
  var13293=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[32]
  var13294=tf.broadcast_to(tf.reshape(var13293, [1]), [32])
  #[32]
  var13295=tf.reshape(var13294, [32])
  #[32]
  var13296=tf.cond(var13286, true_fn=lambda: var13292, false_fn=lambda: var13295)
  #[32]
  var13297=tf.random.uniform([32], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[32]
  var13298=tf.floor(var13297)
  #[32]
  var13299=tf.divide(var13298, var13291)
  #[32]
  var13300=tf.cond(var13286, true_fn=lambda: var13299, false_fn=lambda: var13295)
  #[]
  var13301=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var13302=tf.broadcast_to(tf.reshape(var13301, [1]), [1])
  #[]
  var13303=tf.reshape(var13302, [])
  #[32]
  var13304=tf.one_hot(var13303, axis=0, dtype=tf.float32, depth=32)
  #[32]
  var13305=tf.multiply(var13300, var13304)
  #[1,32]
  var13306=tf.reshape(var13305, [1,32])
  #[32]
  var13307=tf.range(start=0, limit=32, dtype=tf.int32)
  #[32,32]
  var13308=tf.broadcast_to(tf.reshape(var13307, [1,32]), [32,32])
  #[1024]
  var13309=tf.reshape(var13308, [1024])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13310=tf.transpose(var13308, perm=[1,0])
  #[1024]
  var13311=tf.reshape(var13310, [1024])
  #[1024]
  var13312=tf.subtract(var13309, var13311)
  #[1024]
  var13313=tf.broadcast_to(tf.reshape(var13303, [1]), [1024])
  #[1024]
  var13314=tf.math.greater(var13312, var13313)
  #transpose: p = PermSwap; [32,32]
  #[]
  var13315=tf.constant(2, shape=[], dtype=tf.int32)
  #[1]
  var13316=tf.broadcast_to(tf.reshape(var13315, [1]), [1])
  #[]
  var13317=tf.reshape(var13316, [])
  #[]
  var13318=tf.constant(32, shape=[], dtype=tf.int32)
  #[1]
  var13319=tf.broadcast_to(tf.reshape(var13318, [1]), [1])
  #[]
  var13320=tf.reshape(var13319, [])
  #[]
  var13321=tf.multiply(var13317, var13320)
  #[1024]
  var13322=tf.broadcast_to(tf.reshape(var13321, [1]), [1024])
  #[1024]
  var13323=tf.subtract(var13322, var13311)
  #[]
  var13324=tf.constant(3, shape=[], dtype=tf.int32)
  #[1]
  var13325=tf.broadcast_to(tf.reshape(var13324, [1]), [1])
  #[]
  var13326=tf.reshape(var13325, [])
  #[1024]
  var13327=tf.broadcast_to(tf.reshape(var13326, [1]), [1024])
  #[1024]
  var13328=tf.subtract(var13323, var13327)
  #[1024]
  var13329=tf.multiply(var13311, var13328)
  #[1024]
  var13330=tf.broadcast_to(tf.reshape(var13317, [1]), [1024])
  #[1024]
  var13331=tf.math.floordiv(var13329, var13330)
  #[1024]
  var13332=tf.add(var13331, var13309)
  #[]
  var13333=tf.constant(1, shape=[], dtype=tf.int32)
  #[1]
  var13334=tf.broadcast_to(tf.reshape(var13333, [1]), [1])
  #[]
  var13335=tf.reshape(var13334, [])
  #[1024]
  var13336=tf.broadcast_to(tf.reshape(var13335, [1]), [1024])
  #[1024]
  var13337=tf.subtract(var13332, var13336)
  #[]
  var13338=tf.constant(90, shape=[], dtype=tf.int32)
  #[1]
  var13339=tf.broadcast_to(tf.reshape(var13338, [1]), [1])
  #[]
  var13340=tf.reshape(var13339, [])
  #[1024]
  var13341=tf.broadcast_to(tf.reshape(var13340, [1]), [1024])
  #[1024]
  var13342=tf.math.less(var13337, var13341)
  #[1024]
  var13343=tf.math.logical_and(var13314, var13342)
  #[90]
  var13344=tf.random.uniform([90], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[90]
  var13345=tf.floor(var13344)
  #[90]
  var13346=tf.broadcast_to(tf.reshape(var13289, [1]), [90])
  #[90]
  var13347=tf.reshape(var13346, [90])
  #[90]
  var13348=tf.divide(var13345, var13347)
  #[90]
  var13349=tf.broadcast_to(tf.reshape(var13293, [1]), [90])
  #[90]
  var13350=tf.reshape(var13349, [90])
  #[90]
  var13351=tf.cond(var13286, true_fn=lambda: var13348, false_fn=lambda: var13350)
  #[12,90]
  var13352=embs
  #[21]
  var13353=x
  #[1]
  var13354=var13353[0:1]
  #[]
  var13355=tf.reshape(var13354, [])
  #[90]
  var13356=tf.gather(params=var13352, indices=var13355, batch_dims=0, axis=0)
  #[90]
  var13357=tf.multiply(var13351, var13356)
  #[1024]
  var13358=tf.gather(params=var13357, indices=var13337, batch_dims=0, axis=0)
  #[]
  var13359=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var13360=tf.broadcast_to(tf.reshape(var13359, [1]), [1])
  #[]
  var13361=tf.reshape(var13360, [])
  #[1024]
  var13362=tf.broadcast_to(tf.reshape(var13361, [1]), [1024])
  #[1024]
  var13363=tf.where(var13343, var13358, var13362)
  #[32,32]
  var13364=tf.reshape(var13363, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13365=tf.transpose(var13364, perm=[1,0])
  #[32,32]
  var13366=tf.subtract(var13364, var13365)
  #[32,32]
  var13367=tf.linalg.expm(var13366)
  #[1,32]
  var13368=tf.matmul(var13306, var13367)
  #[32]
  var13369=tf.reshape(var13368, [32])
  #[32]
  var13370=tf.multiply(var13296, var13369)
  #[1,32]
  var13371=tf.reshape(var13370, [1,32])
  #[32,12]
  var13372=projection_w
  #[1,12]
  var13373=tf.matmul(var13371, var13372)
  #[12]
  var13374=tf.reshape(var13373, [12])
  #[12]
  var13375=projection_bias
  #[12]
  var13376=tf.add(var13374, var13375)
  #[1,12]
  var13377=tf.reshape(var13376, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13378=tf.multiply(var13300, var13369)
  #[1,32]
  var13379=tf.reshape(var13378, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13380=var13353[1:2]
  #[]
  var13381=tf.reshape(var13380, [])
  #[90]
  var13382=tf.gather(params=var13352, indices=var13381, batch_dims=0, axis=0)
  #[90]
  var13383=tf.multiply(var13351, var13382)
  #[1024]
  var13384=tf.gather(params=var13383, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13385=tf.where(var13343, var13384, var13362)
  #[32,32]
  var13386=tf.reshape(var13385, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13387=tf.transpose(var13386, perm=[1,0])
  #[32,32]
  var13388=tf.subtract(var13386, var13387)
  #[32,32]
  var13389=tf.linalg.expm(var13388)
  #[1,32]
  var13390=tf.matmul(var13379, var13389)
  #[32]
  var13391=tf.reshape(var13390, [32])
  #[32]
  var13392=tf.multiply(var13296, var13391)
  #[1,32]
  var13393=tf.reshape(var13392, [1,32])
  #[1,12]
  var13394=tf.matmul(var13393, var13372)
  #[12]
  var13395=tf.reshape(var13394, [12])
  #[12]
  var13396=tf.add(var13395, var13375)
  #[1,12]
  var13397=tf.reshape(var13396, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13398=tf.multiply(var13300, var13391)
  #[1,32]
  var13399=tf.reshape(var13398, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13400=var13353[2:3]
  #[]
  var13401=tf.reshape(var13400, [])
  #[90]
  var13402=tf.gather(params=var13352, indices=var13401, batch_dims=0, axis=0)
  #[90]
  var13403=tf.multiply(var13351, var13402)
  #[1024]
  var13404=tf.gather(params=var13403, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13405=tf.where(var13343, var13404, var13362)
  #[32,32]
  var13406=tf.reshape(var13405, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13407=tf.transpose(var13406, perm=[1,0])
  #[32,32]
  var13408=tf.subtract(var13406, var13407)
  #[32,32]
  var13409=tf.linalg.expm(var13408)
  #[1,32]
  var13410=tf.matmul(var13399, var13409)
  #[32]
  var13411=tf.reshape(var13410, [32])
  #[32]
  var13412=tf.multiply(var13296, var13411)
  #[1,32]
  var13413=tf.reshape(var13412, [1,32])
  #[1,12]
  var13414=tf.matmul(var13413, var13372)
  #[12]
  var13415=tf.reshape(var13414, [12])
  #[12]
  var13416=tf.add(var13415, var13375)
  #[1,12]
  var13417=tf.reshape(var13416, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13418=tf.multiply(var13300, var13411)
  #[1,32]
  var13419=tf.reshape(var13418, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13420=var13353[3:4]
  #[]
  var13421=tf.reshape(var13420, [])
  #[90]
  var13422=tf.gather(params=var13352, indices=var13421, batch_dims=0, axis=0)
  #[90]
  var13423=tf.multiply(var13351, var13422)
  #[1024]
  var13424=tf.gather(params=var13423, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13425=tf.where(var13343, var13424, var13362)
  #[32,32]
  var13426=tf.reshape(var13425, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13427=tf.transpose(var13426, perm=[1,0])
  #[32,32]
  var13428=tf.subtract(var13426, var13427)
  #[32,32]
  var13429=tf.linalg.expm(var13428)
  #[1,32]
  var13430=tf.matmul(var13419, var13429)
  #[32]
  var13431=tf.reshape(var13430, [32])
  #[32]
  var13432=tf.multiply(var13296, var13431)
  #[1,32]
  var13433=tf.reshape(var13432, [1,32])
  #[1,12]
  var13434=tf.matmul(var13433, var13372)
  #[12]
  var13435=tf.reshape(var13434, [12])
  #[12]
  var13436=tf.add(var13435, var13375)
  #[1,12]
  var13437=tf.reshape(var13436, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13438=tf.multiply(var13300, var13431)
  #[1,32]
  var13439=tf.reshape(var13438, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13440=var13353[4:5]
  #[]
  var13441=tf.reshape(var13440, [])
  #[90]
  var13442=tf.gather(params=var13352, indices=var13441, batch_dims=0, axis=0)
  #[90]
  var13443=tf.multiply(var13351, var13442)
  #[1024]
  var13444=tf.gather(params=var13443, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13445=tf.where(var13343, var13444, var13362)
  #[32,32]
  var13446=tf.reshape(var13445, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13447=tf.transpose(var13446, perm=[1,0])
  #[32,32]
  var13448=tf.subtract(var13446, var13447)
  #[32,32]
  var13449=tf.linalg.expm(var13448)
  #[1,32]
  var13450=tf.matmul(var13439, var13449)
  #[32]
  var13451=tf.reshape(var13450, [32])
  #[32]
  var13452=tf.multiply(var13296, var13451)
  #[1,32]
  var13453=tf.reshape(var13452, [1,32])
  #[1,12]
  var13454=tf.matmul(var13453, var13372)
  #[12]
  var13455=tf.reshape(var13454, [12])
  #[12]
  var13456=tf.add(var13455, var13375)
  #[1,12]
  var13457=tf.reshape(var13456, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13458=tf.multiply(var13300, var13451)
  #[1,32]
  var13459=tf.reshape(var13458, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13460=var13353[5:6]
  #[]
  var13461=tf.reshape(var13460, [])
  #[90]
  var13462=tf.gather(params=var13352, indices=var13461, batch_dims=0, axis=0)
  #[90]
  var13463=tf.multiply(var13351, var13462)
  #[1024]
  var13464=tf.gather(params=var13463, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13465=tf.where(var13343, var13464, var13362)
  #[32,32]
  var13466=tf.reshape(var13465, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13467=tf.transpose(var13466, perm=[1,0])
  #[32,32]
  var13468=tf.subtract(var13466, var13467)
  #[32,32]
  var13469=tf.linalg.expm(var13468)
  #[1,32]
  var13470=tf.matmul(var13459, var13469)
  #[32]
  var13471=tf.reshape(var13470, [32])
  #[32]
  var13472=tf.multiply(var13296, var13471)
  #[1,32]
  var13473=tf.reshape(var13472, [1,32])
  #[1,12]
  var13474=tf.matmul(var13473, var13372)
  #[12]
  var13475=tf.reshape(var13474, [12])
  #[12]
  var13476=tf.add(var13475, var13375)
  #[1,12]
  var13477=tf.reshape(var13476, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13478=tf.multiply(var13300, var13471)
  #[1,32]
  var13479=tf.reshape(var13478, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13480=var13353[6:7]
  #[]
  var13481=tf.reshape(var13480, [])
  #[90]
  var13482=tf.gather(params=var13352, indices=var13481, batch_dims=0, axis=0)
  #[90]
  var13483=tf.multiply(var13351, var13482)
  #[1024]
  var13484=tf.gather(params=var13483, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13485=tf.where(var13343, var13484, var13362)
  #[32,32]
  var13486=tf.reshape(var13485, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13487=tf.transpose(var13486, perm=[1,0])
  #[32,32]
  var13488=tf.subtract(var13486, var13487)
  #[32,32]
  var13489=tf.linalg.expm(var13488)
  #[1,32]
  var13490=tf.matmul(var13479, var13489)
  #[32]
  var13491=tf.reshape(var13490, [32])
  #[32]
  var13492=tf.multiply(var13296, var13491)
  #[1,32]
  var13493=tf.reshape(var13492, [1,32])
  #[1,12]
  var13494=tf.matmul(var13493, var13372)
  #[12]
  var13495=tf.reshape(var13494, [12])
  #[12]
  var13496=tf.add(var13495, var13375)
  #[1,12]
  var13497=tf.reshape(var13496, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13498=tf.multiply(var13300, var13491)
  #[1,32]
  var13499=tf.reshape(var13498, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13500=var13353[7:8]
  #[]
  var13501=tf.reshape(var13500, [])
  #[90]
  var13502=tf.gather(params=var13352, indices=var13501, batch_dims=0, axis=0)
  #[90]
  var13503=tf.multiply(var13351, var13502)
  #[1024]
  var13504=tf.gather(params=var13503, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13505=tf.where(var13343, var13504, var13362)
  #[32,32]
  var13506=tf.reshape(var13505, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13507=tf.transpose(var13506, perm=[1,0])
  #[32,32]
  var13508=tf.subtract(var13506, var13507)
  #[32,32]
  var13509=tf.linalg.expm(var13508)
  #[1,32]
  var13510=tf.matmul(var13499, var13509)
  #[32]
  var13511=tf.reshape(var13510, [32])
  #[32]
  var13512=tf.multiply(var13296, var13511)
  #[1,32]
  var13513=tf.reshape(var13512, [1,32])
  #[1,12]
  var13514=tf.matmul(var13513, var13372)
  #[12]
  var13515=tf.reshape(var13514, [12])
  #[12]
  var13516=tf.add(var13515, var13375)
  #[1,12]
  var13517=tf.reshape(var13516, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13518=tf.multiply(var13300, var13511)
  #[1,32]
  var13519=tf.reshape(var13518, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13520=var13353[8:9]
  #[]
  var13521=tf.reshape(var13520, [])
  #[90]
  var13522=tf.gather(params=var13352, indices=var13521, batch_dims=0, axis=0)
  #[90]
  var13523=tf.multiply(var13351, var13522)
  #[1024]
  var13524=tf.gather(params=var13523, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13525=tf.where(var13343, var13524, var13362)
  #[32,32]
  var13526=tf.reshape(var13525, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13527=tf.transpose(var13526, perm=[1,0])
  #[32,32]
  var13528=tf.subtract(var13526, var13527)
  #[32,32]
  var13529=tf.linalg.expm(var13528)
  #[1,32]
  var13530=tf.matmul(var13519, var13529)
  #[32]
  var13531=tf.reshape(var13530, [32])
  #[32]
  var13532=tf.multiply(var13296, var13531)
  #[1,32]
  var13533=tf.reshape(var13532, [1,32])
  #[1,12]
  var13534=tf.matmul(var13533, var13372)
  #[12]
  var13535=tf.reshape(var13534, [12])
  #[12]
  var13536=tf.add(var13535, var13375)
  #[1,12]
  var13537=tf.reshape(var13536, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13538=tf.multiply(var13300, var13531)
  #[1,32]
  var13539=tf.reshape(var13538, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13540=var13353[9:10]
  #[]
  var13541=tf.reshape(var13540, [])
  #[90]
  var13542=tf.gather(params=var13352, indices=var13541, batch_dims=0, axis=0)
  #[90]
  var13543=tf.multiply(var13351, var13542)
  #[1024]
  var13544=tf.gather(params=var13543, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13545=tf.where(var13343, var13544, var13362)
  #[32,32]
  var13546=tf.reshape(var13545, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13547=tf.transpose(var13546, perm=[1,0])
  #[32,32]
  var13548=tf.subtract(var13546, var13547)
  #[32,32]
  var13549=tf.linalg.expm(var13548)
  #[1,32]
  var13550=tf.matmul(var13539, var13549)
  #[32]
  var13551=tf.reshape(var13550, [32])
  #[32]
  var13552=tf.multiply(var13296, var13551)
  #[1,32]
  var13553=tf.reshape(var13552, [1,32])
  #[1,12]
  var13554=tf.matmul(var13553, var13372)
  #[12]
  var13555=tf.reshape(var13554, [12])
  #[12]
  var13556=tf.add(var13555, var13375)
  #[1,12]
  var13557=tf.reshape(var13556, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13558=tf.multiply(var13300, var13551)
  #[1,32]
  var13559=tf.reshape(var13558, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13560=var13353[10:11]
  #[]
  var13561=tf.reshape(var13560, [])
  #[90]
  var13562=tf.gather(params=var13352, indices=var13561, batch_dims=0, axis=0)
  #[90]
  var13563=tf.multiply(var13351, var13562)
  #[1024]
  var13564=tf.gather(params=var13563, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13565=tf.where(var13343, var13564, var13362)
  #[32,32]
  var13566=tf.reshape(var13565, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13567=tf.transpose(var13566, perm=[1,0])
  #[32,32]
  var13568=tf.subtract(var13566, var13567)
  #[32,32]
  var13569=tf.linalg.expm(var13568)
  #[1,32]
  var13570=tf.matmul(var13559, var13569)
  #[32]
  var13571=tf.reshape(var13570, [32])
  #[32]
  var13572=tf.multiply(var13296, var13571)
  #[1,32]
  var13573=tf.reshape(var13572, [1,32])
  #[1,12]
  var13574=tf.matmul(var13573, var13372)
  #[12]
  var13575=tf.reshape(var13574, [12])
  #[12]
  var13576=tf.add(var13575, var13375)
  #[1,12]
  var13577=tf.reshape(var13576, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13578=tf.multiply(var13300, var13571)
  #[1,32]
  var13579=tf.reshape(var13578, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13580=var13353[11:12]
  #[]
  var13581=tf.reshape(var13580, [])
  #[90]
  var13582=tf.gather(params=var13352, indices=var13581, batch_dims=0, axis=0)
  #[90]
  var13583=tf.multiply(var13351, var13582)
  #[1024]
  var13584=tf.gather(params=var13583, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13585=tf.where(var13343, var13584, var13362)
  #[32,32]
  var13586=tf.reshape(var13585, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13587=tf.transpose(var13586, perm=[1,0])
  #[32,32]
  var13588=tf.subtract(var13586, var13587)
  #[32,32]
  var13589=tf.linalg.expm(var13588)
  #[1,32]
  var13590=tf.matmul(var13579, var13589)
  #[32]
  var13591=tf.reshape(var13590, [32])
  #[32]
  var13592=tf.multiply(var13296, var13591)
  #[1,32]
  var13593=tf.reshape(var13592, [1,32])
  #[1,12]
  var13594=tf.matmul(var13593, var13372)
  #[12]
  var13595=tf.reshape(var13594, [12])
  #[12]
  var13596=tf.add(var13595, var13375)
  #[1,12]
  var13597=tf.reshape(var13596, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13598=tf.multiply(var13300, var13591)
  #[1,32]
  var13599=tf.reshape(var13598, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13600=var13353[12:13]
  #[]
  var13601=tf.reshape(var13600, [])
  #[90]
  var13602=tf.gather(params=var13352, indices=var13601, batch_dims=0, axis=0)
  #[90]
  var13603=tf.multiply(var13351, var13602)
  #[1024]
  var13604=tf.gather(params=var13603, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13605=tf.where(var13343, var13604, var13362)
  #[32,32]
  var13606=tf.reshape(var13605, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13607=tf.transpose(var13606, perm=[1,0])
  #[32,32]
  var13608=tf.subtract(var13606, var13607)
  #[32,32]
  var13609=tf.linalg.expm(var13608)
  #[1,32]
  var13610=tf.matmul(var13599, var13609)
  #[32]
  var13611=tf.reshape(var13610, [32])
  #[32]
  var13612=tf.multiply(var13296, var13611)
  #[1,32]
  var13613=tf.reshape(var13612, [1,32])
  #[1,12]
  var13614=tf.matmul(var13613, var13372)
  #[12]
  var13615=tf.reshape(var13614, [12])
  #[12]
  var13616=tf.add(var13615, var13375)
  #[1,12]
  var13617=tf.reshape(var13616, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13618=tf.multiply(var13300, var13611)
  #[1,32]
  var13619=tf.reshape(var13618, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13620=var13353[13:14]
  #[]
  var13621=tf.reshape(var13620, [])
  #[90]
  var13622=tf.gather(params=var13352, indices=var13621, batch_dims=0, axis=0)
  #[90]
  var13623=tf.multiply(var13351, var13622)
  #[1024]
  var13624=tf.gather(params=var13623, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13625=tf.where(var13343, var13624, var13362)
  #[32,32]
  var13626=tf.reshape(var13625, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13627=tf.transpose(var13626, perm=[1,0])
  #[32,32]
  var13628=tf.subtract(var13626, var13627)
  #[32,32]
  var13629=tf.linalg.expm(var13628)
  #[1,32]
  var13630=tf.matmul(var13619, var13629)
  #[32]
  var13631=tf.reshape(var13630, [32])
  #[32]
  var13632=tf.multiply(var13296, var13631)
  #[1,32]
  var13633=tf.reshape(var13632, [1,32])
  #[1,12]
  var13634=tf.matmul(var13633, var13372)
  #[12]
  var13635=tf.reshape(var13634, [12])
  #[12]
  var13636=tf.add(var13635, var13375)
  #[1,12]
  var13637=tf.reshape(var13636, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13638=tf.multiply(var13300, var13631)
  #[1,32]
  var13639=tf.reshape(var13638, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13640=var13353[14:15]
  #[]
  var13641=tf.reshape(var13640, [])
  #[90]
  var13642=tf.gather(params=var13352, indices=var13641, batch_dims=0, axis=0)
  #[90]
  var13643=tf.multiply(var13351, var13642)
  #[1024]
  var13644=tf.gather(params=var13643, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13645=tf.where(var13343, var13644, var13362)
  #[32,32]
  var13646=tf.reshape(var13645, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13647=tf.transpose(var13646, perm=[1,0])
  #[32,32]
  var13648=tf.subtract(var13646, var13647)
  #[32,32]
  var13649=tf.linalg.expm(var13648)
  #[1,32]
  var13650=tf.matmul(var13639, var13649)
  #[32]
  var13651=tf.reshape(var13650, [32])
  #[32]
  var13652=tf.multiply(var13296, var13651)
  #[1,32]
  var13653=tf.reshape(var13652, [1,32])
  #[1,12]
  var13654=tf.matmul(var13653, var13372)
  #[12]
  var13655=tf.reshape(var13654, [12])
  #[12]
  var13656=tf.add(var13655, var13375)
  #[1,12]
  var13657=tf.reshape(var13656, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13658=tf.multiply(var13300, var13651)
  #[1,32]
  var13659=tf.reshape(var13658, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13660=var13353[15:16]
  #[]
  var13661=tf.reshape(var13660, [])
  #[90]
  var13662=tf.gather(params=var13352, indices=var13661, batch_dims=0, axis=0)
  #[90]
  var13663=tf.multiply(var13351, var13662)
  #[1024]
  var13664=tf.gather(params=var13663, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13665=tf.where(var13343, var13664, var13362)
  #[32,32]
  var13666=tf.reshape(var13665, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13667=tf.transpose(var13666, perm=[1,0])
  #[32,32]
  var13668=tf.subtract(var13666, var13667)
  #[32,32]
  var13669=tf.linalg.expm(var13668)
  #[1,32]
  var13670=tf.matmul(var13659, var13669)
  #[32]
  var13671=tf.reshape(var13670, [32])
  #[32]
  var13672=tf.multiply(var13296, var13671)
  #[1,32]
  var13673=tf.reshape(var13672, [1,32])
  #[1,12]
  var13674=tf.matmul(var13673, var13372)
  #[12]
  var13675=tf.reshape(var13674, [12])
  #[12]
  var13676=tf.add(var13675, var13375)
  #[1,12]
  var13677=tf.reshape(var13676, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13678=tf.multiply(var13300, var13671)
  #[1,32]
  var13679=tf.reshape(var13678, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13680=var13353[16:17]
  #[]
  var13681=tf.reshape(var13680, [])
  #[90]
  var13682=tf.gather(params=var13352, indices=var13681, batch_dims=0, axis=0)
  #[90]
  var13683=tf.multiply(var13351, var13682)
  #[1024]
  var13684=tf.gather(params=var13683, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13685=tf.where(var13343, var13684, var13362)
  #[32,32]
  var13686=tf.reshape(var13685, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13687=tf.transpose(var13686, perm=[1,0])
  #[32,32]
  var13688=tf.subtract(var13686, var13687)
  #[32,32]
  var13689=tf.linalg.expm(var13688)
  #[1,32]
  var13690=tf.matmul(var13679, var13689)
  #[32]
  var13691=tf.reshape(var13690, [32])
  #[32]
  var13692=tf.multiply(var13296, var13691)
  #[1,32]
  var13693=tf.reshape(var13692, [1,32])
  #[1,12]
  var13694=tf.matmul(var13693, var13372)
  #[12]
  var13695=tf.reshape(var13694, [12])
  #[12]
  var13696=tf.add(var13695, var13375)
  #[1,12]
  var13697=tf.reshape(var13696, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13698=tf.multiply(var13300, var13691)
  #[1,32]
  var13699=tf.reshape(var13698, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13700=var13353[17:18]
  #[]
  var13701=tf.reshape(var13700, [])
  #[90]
  var13702=tf.gather(params=var13352, indices=var13701, batch_dims=0, axis=0)
  #[90]
  var13703=tf.multiply(var13351, var13702)
  #[1024]
  var13704=tf.gather(params=var13703, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13705=tf.where(var13343, var13704, var13362)
  #[32,32]
  var13706=tf.reshape(var13705, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13707=tf.transpose(var13706, perm=[1,0])
  #[32,32]
  var13708=tf.subtract(var13706, var13707)
  #[32,32]
  var13709=tf.linalg.expm(var13708)
  #[1,32]
  var13710=tf.matmul(var13699, var13709)
  #[32]
  var13711=tf.reshape(var13710, [32])
  #[32]
  var13712=tf.multiply(var13296, var13711)
  #[1,32]
  var13713=tf.reshape(var13712, [1,32])
  #[1,12]
  var13714=tf.matmul(var13713, var13372)
  #[12]
  var13715=tf.reshape(var13714, [12])
  #[12]
  var13716=tf.add(var13715, var13375)
  #[1,12]
  var13717=tf.reshape(var13716, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13718=tf.multiply(var13300, var13711)
  #[1,32]
  var13719=tf.reshape(var13718, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13720=var13353[18:19]
  #[]
  var13721=tf.reshape(var13720, [])
  #[90]
  var13722=tf.gather(params=var13352, indices=var13721, batch_dims=0, axis=0)
  #[90]
  var13723=tf.multiply(var13351, var13722)
  #[1024]
  var13724=tf.gather(params=var13723, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13725=tf.where(var13343, var13724, var13362)
  #[32,32]
  var13726=tf.reshape(var13725, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13727=tf.transpose(var13726, perm=[1,0])
  #[32,32]
  var13728=tf.subtract(var13726, var13727)
  #[32,32]
  var13729=tf.linalg.expm(var13728)
  #[1,32]
  var13730=tf.matmul(var13719, var13729)
  #[32]
  var13731=tf.reshape(var13730, [32])
  #[32]
  var13732=tf.multiply(var13296, var13731)
  #[1,32]
  var13733=tf.reshape(var13732, [1,32])
  #[1,12]
  var13734=tf.matmul(var13733, var13372)
  #[12]
  var13735=tf.reshape(var13734, [12])
  #[12]
  var13736=tf.add(var13735, var13375)
  #[1,12]
  var13737=tf.reshape(var13736, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13738=tf.multiply(var13300, var13731)
  #[1,32]
  var13739=tf.reshape(var13738, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13740=var13353[19:20]
  #[]
  var13741=tf.reshape(var13740, [])
  #[90]
  var13742=tf.gather(params=var13352, indices=var13741, batch_dims=0, axis=0)
  #[90]
  var13743=tf.multiply(var13351, var13742)
  #[1024]
  var13744=tf.gather(params=var13743, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13745=tf.where(var13343, var13744, var13362)
  #[32,32]
  var13746=tf.reshape(var13745, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13747=tf.transpose(var13746, perm=[1,0])
  #[32,32]
  var13748=tf.subtract(var13746, var13747)
  #[32,32]
  var13749=tf.linalg.expm(var13748)
  #[1,32]
  var13750=tf.matmul(var13739, var13749)
  #[32]
  var13751=tf.reshape(var13750, [32])
  #[32]
  var13752=tf.multiply(var13296, var13751)
  #[1,32]
  var13753=tf.reshape(var13752, [1,32])
  #[1,12]
  var13754=tf.matmul(var13753, var13372)
  #[12]
  var13755=tf.reshape(var13754, [12])
  #[12]
  var13756=tf.add(var13755, var13375)
  #[1,12]
  var13757=tf.reshape(var13756, [1,12])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13758=tf.multiply(var13300, var13751)
  #[1,32]
  var13759=tf.reshape(var13758, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13760=var13353[20:21]
  #[]
  var13761=tf.reshape(var13760, [])
  #[90]
  var13762=tf.gather(params=var13352, indices=var13761, batch_dims=0, axis=0)
  #[90]
  var13763=tf.multiply(var13351, var13762)
  #[1024]
  var13764=tf.gather(params=var13763, indices=var13337, batch_dims=0, axis=0)
  #[1024]
  var13765=tf.where(var13343, var13764, var13362)
  #[32,32]
  var13766=tf.reshape(var13765, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13767=tf.transpose(var13766, perm=[1,0])
  #[32,32]
  var13768=tf.subtract(var13766, var13767)
  #[32,32]
  var13769=tf.linalg.expm(var13768)
  #[1,32]
  var13770=tf.matmul(var13759, var13769)
  #[32]
  var13771=tf.reshape(var13770, [32])
  #[32]
  var13772=tf.multiply(var13296, var13771)
  #[1,32]
  var13773=tf.reshape(var13772, [1,32])
  #[1,12]
  var13774=tf.matmul(var13773, var13372)
  #[12]
  var13775=tf.reshape(var13774, [12])
  #[12]
  var13776=tf.add(var13775, var13375)
  #[1,12]
  var13777=tf.reshape(var13776, [1,12])
  #[21,12]
  var13778=tf.concat([var13377
                     ,var13397
                     ,var13417
                     ,var13437
                     ,var13457
                     ,var13477
                     ,var13497
                     ,var13517
                     ,var13537
                     ,var13557
                     ,var13577
                     ,var13597
                     ,var13617
                     ,var13637
                     ,var13657
                     ,var13677
                     ,var13697
                     ,var13717
                     ,var13737
                     ,var13757
                     ,var13777],
                     axis=0)
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[21]
  var13779=tf.argmax(var13778, axis=1, output_type=tf.int32)
  return {"pred":var13778,"y":var13779}
probePreds = {"function":probePreds_fn
             ,"batched":False
             ,"placeholders":{"x":{"shape":[21],"dtype":tf.int32}}}
@tf.function
def probeEmbs_fn(training_placeholder, embs, projection_w, projection_bias, wordIdx):
  
  #[32]
  var13780=tf.range(start=0, limit=32, dtype=tf.int32)
  #[32,32]
  var13781=tf.broadcast_to(tf.reshape(var13780, [1,32]), [32,32])
  #[1024]
  var13782=tf.reshape(var13781, [1024])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13783=tf.transpose(var13781, perm=[1,0])
  #[1024]
  var13784=tf.reshape(var13783, [1024])
  #[1024]
  var13785=tf.subtract(var13782, var13784)
  #[]
  var13786=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var13787=tf.broadcast_to(tf.reshape(var13786, [1]), [1])
  #[]
  var13788=tf.reshape(var13787, [])
  #[1024]
  var13789=tf.broadcast_to(tf.reshape(var13788, [1]), [1024])
  #[1024]
  var13790=tf.math.greater(var13785, var13789)
  #transpose: p = PermSwap; [32,32]
  #[]
  var13791=tf.constant(2, shape=[], dtype=tf.int32)
  #[1]
  var13792=tf.broadcast_to(tf.reshape(var13791, [1]), [1])
  #[]
  var13793=tf.reshape(var13792, [])
  #[]
  var13794=tf.constant(32, shape=[], dtype=tf.int32)
  #[1]
  var13795=tf.broadcast_to(tf.reshape(var13794, [1]), [1])
  #[]
  var13796=tf.reshape(var13795, [])
  #[]
  var13797=tf.multiply(var13793, var13796)
  #[1024]
  var13798=tf.broadcast_to(tf.reshape(var13797, [1]), [1024])
  #[1024]
  var13799=tf.subtract(var13798, var13784)
  #[]
  var13800=tf.constant(3, shape=[], dtype=tf.int32)
  #[1]
  var13801=tf.broadcast_to(tf.reshape(var13800, [1]), [1])
  #[]
  var13802=tf.reshape(var13801, [])
  #[1024]
  var13803=tf.broadcast_to(tf.reshape(var13802, [1]), [1024])
  #[1024]
  var13804=tf.subtract(var13799, var13803)
  #[1024]
  var13805=tf.multiply(var13784, var13804)
  #[1024]
  var13806=tf.broadcast_to(tf.reshape(var13793, [1]), [1024])
  #[1024]
  var13807=tf.math.floordiv(var13805, var13806)
  #[1024]
  var13808=tf.add(var13807, var13782)
  #[]
  var13809=tf.constant(1, shape=[], dtype=tf.int32)
  #[1]
  var13810=tf.broadcast_to(tf.reshape(var13809, [1]), [1])
  #[]
  var13811=tf.reshape(var13810, [])
  #[1024]
  var13812=tf.broadcast_to(tf.reshape(var13811, [1]), [1024])
  #[1024]
  var13813=tf.subtract(var13808, var13812)
  #[]
  var13814=tf.constant(90, shape=[], dtype=tf.int32)
  #[1]
  var13815=tf.broadcast_to(tf.reshape(var13814, [1]), [1])
  #[]
  var13816=tf.reshape(var13815, [])
  #[1024]
  var13817=tf.broadcast_to(tf.reshape(var13816, [1]), [1024])
  #[1024]
  var13818=tf.math.less(var13813, var13817)
  #[1024]
  var13819=tf.math.logical_and(var13790, var13818)
  #[12,90]
  var13820=embs
  #[]
  var13821=wordIdx
  #[90]
  var13822=tf.gather(params=var13820, indices=var13821, batch_dims=0, axis=0)
  #[1024]
  var13823=tf.gather(params=var13822, indices=var13813, batch_dims=0, axis=0)
  #[]
  var13824=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var13825=tf.broadcast_to(tf.reshape(var13824, [1]), [1])
  #[]
  var13826=tf.reshape(var13825, [])
  #[1024]
  var13827=tf.broadcast_to(tf.reshape(var13826, [1]), [1024])
  #[1024]
  var13828=tf.where(var13819, var13823, var13827)
  #[32,32]
  var13829=tf.reshape(var13828, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13830=tf.transpose(var13829, perm=[1,0])
  #[32,32]
  var13831=tf.subtract(var13829, var13830)
  return {"embsAntiHermitian":var13831}
probeEmbs = {"function":probeEmbs_fn
            ,"batched":False
            ,"placeholders":{"wordIdx":{"shape":[],"dtype":tf.int32}}}