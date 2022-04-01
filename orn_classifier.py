
import tensorflow as tf
def mkModel():
  
  #[50050,150]
  var12345=tf.random.uniform(
             [50050,150], minval=-5.0e-2, maxval=5.0e-2, dtype=tf.float32) # 0
  var12346=tf.Variable(name="embs", trainable=True, initial_value=var12345)
  #[32,2]
  var12347=tf.random.uniform(
             [32,2], minval=-0.42008403, maxval=0.42008403, dtype=tf.float32) # 4
  var12348=tf.Variable(name="dense_w", trainable=True, initial_value=var12347)
  #[2]
  var12349=tf.random.truncated_normal([2], stddev=0.1, dtype=tf.float32) # 5
  var12350=tf.Variable(name="dense_bias", trainable=True, initial_value=var12349)
  return {"batch_size":512
         ,"parameters":[var12346,var12348,var12350]
         ,"paramsdict":{"embs":var12346,"dense_w":var12348,"dense_bias":var12350}}
@tf.function
def runModel_fn(training_placeholder, embs, dense_w, dense_bias, x, yIndex, y):
  
  #[512]
  var12351=y
  #[512,2]
  var12352=tf.one_hot(var12351, axis=1, dtype=tf.float32, depth=2)
  #[]
  var12353=training_placeholder
  #[512,32]
  var12354=tf.random.uniform([512,32], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[512,32]
  var12355=tf.floor(var12354)
  #[]
  var12356=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[32]
  var12357=tf.broadcast_to(tf.reshape(var12356, [1]), [32])
  #[32]
  var12358=tf.reshape(var12357, [32])
  #[512,32]
  var12359=tf.broadcast_to(tf.reshape(var12358, [1,32]), [512,32])
  #[512,32]
  var12360=tf.divide(var12355, var12359)
  #[]
  var12361=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[32]
  var12362=tf.broadcast_to(tf.reshape(var12361, [1]), [32])
  #[32]
  var12363=tf.reshape(var12362, [32])
  #[512,32]
  var12364=tf.broadcast_to(tf.reshape(var12363, [1,32]), [512,32])
  #[512,32]
  var12365=tf.cond(var12353, true_fn=lambda: var12360, false_fn=lambda: var12364)
  #[512,32]
  var12366=tf.random.uniform([512,32], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[512,32]
  var12367=tf.floor(var12366)
  #[512,32]
  var12368=tf.divide(var12367, var12359)
  #[512,32]
  var12369=tf.cond(var12353, true_fn=lambda: var12368, false_fn=lambda: var12364)
  #[]
  var12370=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var12371=tf.broadcast_to(tf.reshape(var12370, [1]), [1])
  #[]
  var12372=tf.reshape(var12371, [])
  #[32]
  var12373=tf.one_hot(var12372, axis=0, dtype=tf.float32, depth=32)
  #[512,32]
  var12374=tf.broadcast_to(tf.reshape(var12373, [1,32]), [512,32])
  #[512,32]
  var12375=tf.multiply(var12369, var12374)
  #[512,1,32]
  var12376=tf.reshape(var12375, [512,1,32])
  #[32]
  var12377=tf.range(start=0, limit=32, dtype=tf.int32)
  #[32,32]
  var12378=tf.broadcast_to(tf.reshape(var12377, [1,32]), [32,32])
  #[1024]
  var12379=tf.reshape(var12378, [1024])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var12380=tf.transpose(var12378, perm=[1,0])
  #[1024]
  var12381=tf.reshape(var12380, [1024])
  #[1024]
  var12382=tf.subtract(var12379, var12381)
  #[1024]
  var12383=tf.broadcast_to(tf.reshape(var12372, [1]), [1024])
  #[1024]
  var12384=tf.math.greater(var12382, var12383)
  #[]
  var12385=tf.constant(2, shape=[], dtype=tf.int32)
  #[1]
  var12386=tf.broadcast_to(tf.reshape(var12385, [1]), [1])
  #[]
  var12387=tf.reshape(var12386, [])
  #[]
  var12388=tf.constant(32, shape=[], dtype=tf.int32)
  #[1]
  var12389=tf.broadcast_to(tf.reshape(var12388, [1]), [1])
  #[]
  var12390=tf.reshape(var12389, [])
  #[]
  var12391=tf.multiply(var12387, var12390)
  #[1024]
  var12392=tf.broadcast_to(tf.reshape(var12391, [1]), [1024])
  #[1024]
  var12393=tf.subtract(var12392, var12381)
  #[]
  var12394=tf.constant(3, shape=[], dtype=tf.int32)
  #[1]
  var12395=tf.broadcast_to(tf.reshape(var12394, [1]), [1])
  #[]
  var12396=tf.reshape(var12395, [])
  #[1024]
  var12397=tf.broadcast_to(tf.reshape(var12396, [1]), [1024])
  #[1024]
  var12398=tf.subtract(var12393, var12397)
  #[1024]
  var12399=tf.multiply(var12381, var12398)
  #[1024]
  var12400=tf.broadcast_to(tf.reshape(var12387, [1]), [1024])
  #[1024]
  var12401=tf.math.floordiv(var12399, var12400)
  #[1024]
  var12402=tf.add(var12401, var12379)
  #[]
  var12403=tf.constant(1, shape=[], dtype=tf.int32)
  #[1]
  var12404=tf.broadcast_to(tf.reshape(var12403, [1]), [1])
  #[]
  var12405=tf.reshape(var12404, [])
  #[1024]
  var12406=tf.broadcast_to(tf.reshape(var12405, [1]), [1024])
  #[1024]
  var12407=tf.subtract(var12402, var12406)
  #[]
  var12408=tf.constant(150, shape=[], dtype=tf.int32)
  #[1]
  var12409=tf.broadcast_to(tf.reshape(var12408, [1]), [1])
  #[]
  var12410=tf.reshape(var12409, [])
  #[1024]
  var12411=tf.broadcast_to(tf.reshape(var12410, [1]), [1024])
  #[1024]
  var12412=tf.math.less(var12407, var12411)
  #[1024]
  var12413=tf.math.logical_and(var12384, var12412)
  #[512,1024]
  var12414=tf.broadcast_to(tf.reshape(var12413, [1,1024]), [512,1024])
  #[512,150]
  var12415=tf.random.uniform([512,150], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[512,150]
  var12416=tf.floor(var12415)
  #[150]
  var12417=tf.broadcast_to(tf.reshape(var12356, [1]), [150])
  #[150]
  var12418=tf.reshape(var12417, [150])
  #[512,150]
  var12419=tf.broadcast_to(tf.reshape(var12418, [1,150]), [512,150])
  #[512,150]
  var12420=tf.divide(var12416, var12419)
  #[150]
  var12421=tf.broadcast_to(tf.reshape(var12361, [1]), [150])
  #[150]
  var12422=tf.reshape(var12421, [150])
  #[512,150]
  var12423=tf.broadcast_to(tf.reshape(var12422, [1,150]), [512,150])
  #[512,150]
  var12424=tf.cond(var12353, true_fn=lambda: var12420, false_fn=lambda: var12423)
  #[50050,150]
  var12425=embs
  #[512,50]
  var12426=x
  #[512,1]
  var12427=var12426[:,0:1]
  #[512]
  var12428=tf.reshape(var12427, [512])
  #[512,150]
  var12429=tf.gather(params=var12425, indices=var12428, batch_dims=0, axis=0)
  #[512,150]
  var12430=tf.multiply(var12424, var12429)
  #[512,1024]
  var12431=tf.broadcast_to(tf.reshape(var12407, [1,1024]), [512,1024])
  #[512,1024]
  var12432=tf.gather(params=var12430, indices=var12431, batch_dims=1, axis=1)
  #[]
  var12433=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var12434=tf.broadcast_to(tf.reshape(var12433, [1]), [1])
  #[]
  var12435=tf.reshape(var12434, [])
  #[1024]
  var12436=tf.broadcast_to(tf.reshape(var12435, [1]), [1024])
  #[512,1024]
  var12437=tf.broadcast_to(tf.reshape(var12436, [1,1024]), [512,1024])
  #[512,1024]
  var12438=tf.where(var12414, var12432, var12437)
  #[512,32,32]
  var12439=tf.reshape(var12438, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12440=tf.transpose(var12439, perm=[0,2,1])
  #[512,32,32]
  var12441=tf.subtract(var12439, var12440)
  #[512,32,32]
  var12442=tf.linalg.expm(var12441)
  #[512,1,32]
  var12443=tf.matmul(var12376, var12442)
  #[512,32]
  var12444=tf.reshape(var12443, [512,32])
  #[512,32]
  var12445=tf.multiply(var12365, var12444)
  #[512,32]
  var12446=tf.reshape(var12445, [512,32])
  #[32,2]
  var12447=dense_w
  #[512,2]
  var12448=tf.matmul(var12446, var12447)
  #[512,2]
  var12449=tf.reshape(var12448, [512,2])
  #[2]
  var12450=dense_bias
  #[512,2]
  var12451=tf.broadcast_to(tf.reshape(var12450, [1,2]), [512,2])
  #[512,2]
  var12452=tf.add(var12449, var12451)
  #[512,1,2]
  var12453=tf.reshape(var12452, [512,1,2])
  #[512,32]
  var12454=tf.multiply(var12369, var12444)
  #[512,1,32]
  var12455=tf.reshape(var12454, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12456=var12426[:,1:2]
  #[512]
  var12457=tf.reshape(var12456, [512])
  #[512,150]
  var12458=tf.gather(params=var12425, indices=var12457, batch_dims=0, axis=0)
  #[512,150]
  var12459=tf.multiply(var12424, var12458)
  #[512,1024]
  var12460=tf.gather(params=var12459, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12461=tf.where(var12414, var12460, var12437)
  #[512,32,32]
  var12462=tf.reshape(var12461, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12463=tf.transpose(var12462, perm=[0,2,1])
  #[512,32,32]
  var12464=tf.subtract(var12462, var12463)
  #[512,32,32]
  var12465=tf.linalg.expm(var12464)
  #[512,1,32]
  var12466=tf.matmul(var12455, var12465)
  #[512,32]
  var12467=tf.reshape(var12466, [512,32])
  #[512,32]
  var12468=tf.multiply(var12365, var12467)
  #[512,32]
  var12469=tf.reshape(var12468, [512,32])
  #[512,2]
  var12470=tf.matmul(var12469, var12447)
  #[512,2]
  var12471=tf.reshape(var12470, [512,2])
  #[512,2]
  var12472=tf.add(var12471, var12451)
  #[512,1,2]
  var12473=tf.reshape(var12472, [512,1,2])
  #[512,32]
  var12474=tf.multiply(var12369, var12467)
  #[512,1,32]
  var12475=tf.reshape(var12474, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12476=var12426[:,2:3]
  #[512]
  var12477=tf.reshape(var12476, [512])
  #[512,150]
  var12478=tf.gather(params=var12425, indices=var12477, batch_dims=0, axis=0)
  #[512,150]
  var12479=tf.multiply(var12424, var12478)
  #[512,1024]
  var12480=tf.gather(params=var12479, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12481=tf.where(var12414, var12480, var12437)
  #[512,32,32]
  var12482=tf.reshape(var12481, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12483=tf.transpose(var12482, perm=[0,2,1])
  #[512,32,32]
  var12484=tf.subtract(var12482, var12483)
  #[512,32,32]
  var12485=tf.linalg.expm(var12484)
  #[512,1,32]
  var12486=tf.matmul(var12475, var12485)
  #[512,32]
  var12487=tf.reshape(var12486, [512,32])
  #[512,32]
  var12488=tf.multiply(var12365, var12487)
  #[512,32]
  var12489=tf.reshape(var12488, [512,32])
  #[512,2]
  var12490=tf.matmul(var12489, var12447)
  #[512,2]
  var12491=tf.reshape(var12490, [512,2])
  #[512,2]
  var12492=tf.add(var12491, var12451)
  #[512,1,2]
  var12493=tf.reshape(var12492, [512,1,2])
  #[512,32]
  var12494=tf.multiply(var12369, var12487)
  #[512,1,32]
  var12495=tf.reshape(var12494, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12496=var12426[:,3:4]
  #[512]
  var12497=tf.reshape(var12496, [512])
  #[512,150]
  var12498=tf.gather(params=var12425, indices=var12497, batch_dims=0, axis=0)
  #[512,150]
  var12499=tf.multiply(var12424, var12498)
  #[512,1024]
  var12500=tf.gather(params=var12499, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12501=tf.where(var12414, var12500, var12437)
  #[512,32,32]
  var12502=tf.reshape(var12501, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12503=tf.transpose(var12502, perm=[0,2,1])
  #[512,32,32]
  var12504=tf.subtract(var12502, var12503)
  #[512,32,32]
  var12505=tf.linalg.expm(var12504)
  #[512,1,32]
  var12506=tf.matmul(var12495, var12505)
  #[512,32]
  var12507=tf.reshape(var12506, [512,32])
  #[512,32]
  var12508=tf.multiply(var12365, var12507)
  #[512,32]
  var12509=tf.reshape(var12508, [512,32])
  #[512,2]
  var12510=tf.matmul(var12509, var12447)
  #[512,2]
  var12511=tf.reshape(var12510, [512,2])
  #[512,2]
  var12512=tf.add(var12511, var12451)
  #[512,1,2]
  var12513=tf.reshape(var12512, [512,1,2])
  #[512,32]
  var12514=tf.multiply(var12369, var12507)
  #[512,1,32]
  var12515=tf.reshape(var12514, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12516=var12426[:,4:5]
  #[512]
  var12517=tf.reshape(var12516, [512])
  #[512,150]
  var12518=tf.gather(params=var12425, indices=var12517, batch_dims=0, axis=0)
  #[512,150]
  var12519=tf.multiply(var12424, var12518)
  #[512,1024]
  var12520=tf.gather(params=var12519, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12521=tf.where(var12414, var12520, var12437)
  #[512,32,32]
  var12522=tf.reshape(var12521, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12523=tf.transpose(var12522, perm=[0,2,1])
  #[512,32,32]
  var12524=tf.subtract(var12522, var12523)
  #[512,32,32]
  var12525=tf.linalg.expm(var12524)
  #[512,1,32]
  var12526=tf.matmul(var12515, var12525)
  #[512,32]
  var12527=tf.reshape(var12526, [512,32])
  #[512,32]
  var12528=tf.multiply(var12365, var12527)
  #[512,32]
  var12529=tf.reshape(var12528, [512,32])
  #[512,2]
  var12530=tf.matmul(var12529, var12447)
  #[512,2]
  var12531=tf.reshape(var12530, [512,2])
  #[512,2]
  var12532=tf.add(var12531, var12451)
  #[512,1,2]
  var12533=tf.reshape(var12532, [512,1,2])
  #[512,32]
  var12534=tf.multiply(var12369, var12527)
  #[512,1,32]
  var12535=tf.reshape(var12534, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12536=var12426[:,5:6]
  #[512]
  var12537=tf.reshape(var12536, [512])
  #[512,150]
  var12538=tf.gather(params=var12425, indices=var12537, batch_dims=0, axis=0)
  #[512,150]
  var12539=tf.multiply(var12424, var12538)
  #[512,1024]
  var12540=tf.gather(params=var12539, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12541=tf.where(var12414, var12540, var12437)
  #[512,32,32]
  var12542=tf.reshape(var12541, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12543=tf.transpose(var12542, perm=[0,2,1])
  #[512,32,32]
  var12544=tf.subtract(var12542, var12543)
  #[512,32,32]
  var12545=tf.linalg.expm(var12544)
  #[512,1,32]
  var12546=tf.matmul(var12535, var12545)
  #[512,32]
  var12547=tf.reshape(var12546, [512,32])
  #[512,32]
  var12548=tf.multiply(var12365, var12547)
  #[512,32]
  var12549=tf.reshape(var12548, [512,32])
  #[512,2]
  var12550=tf.matmul(var12549, var12447)
  #[512,2]
  var12551=tf.reshape(var12550, [512,2])
  #[512,2]
  var12552=tf.add(var12551, var12451)
  #[512,1,2]
  var12553=tf.reshape(var12552, [512,1,2])
  #[512,32]
  var12554=tf.multiply(var12369, var12547)
  #[512,1,32]
  var12555=tf.reshape(var12554, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12556=var12426[:,6:7]
  #[512]
  var12557=tf.reshape(var12556, [512])
  #[512,150]
  var12558=tf.gather(params=var12425, indices=var12557, batch_dims=0, axis=0)
  #[512,150]
  var12559=tf.multiply(var12424, var12558)
  #[512,1024]
  var12560=tf.gather(params=var12559, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12561=tf.where(var12414, var12560, var12437)
  #[512,32,32]
  var12562=tf.reshape(var12561, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12563=tf.transpose(var12562, perm=[0,2,1])
  #[512,32,32]
  var12564=tf.subtract(var12562, var12563)
  #[512,32,32]
  var12565=tf.linalg.expm(var12564)
  #[512,1,32]
  var12566=tf.matmul(var12555, var12565)
  #[512,32]
  var12567=tf.reshape(var12566, [512,32])
  #[512,32]
  var12568=tf.multiply(var12365, var12567)
  #[512,32]
  var12569=tf.reshape(var12568, [512,32])
  #[512,2]
  var12570=tf.matmul(var12569, var12447)
  #[512,2]
  var12571=tf.reshape(var12570, [512,2])
  #[512,2]
  var12572=tf.add(var12571, var12451)
  #[512,1,2]
  var12573=tf.reshape(var12572, [512,1,2])
  #[512,32]
  var12574=tf.multiply(var12369, var12567)
  #[512,1,32]
  var12575=tf.reshape(var12574, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12576=var12426[:,7:8]
  #[512]
  var12577=tf.reshape(var12576, [512])
  #[512,150]
  var12578=tf.gather(params=var12425, indices=var12577, batch_dims=0, axis=0)
  #[512,150]
  var12579=tf.multiply(var12424, var12578)
  #[512,1024]
  var12580=tf.gather(params=var12579, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12581=tf.where(var12414, var12580, var12437)
  #[512,32,32]
  var12582=tf.reshape(var12581, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12583=tf.transpose(var12582, perm=[0,2,1])
  #[512,32,32]
  var12584=tf.subtract(var12582, var12583)
  #[512,32,32]
  var12585=tf.linalg.expm(var12584)
  #[512,1,32]
  var12586=tf.matmul(var12575, var12585)
  #[512,32]
  var12587=tf.reshape(var12586, [512,32])
  #[512,32]
  var12588=tf.multiply(var12365, var12587)
  #[512,32]
  var12589=tf.reshape(var12588, [512,32])
  #[512,2]
  var12590=tf.matmul(var12589, var12447)
  #[512,2]
  var12591=tf.reshape(var12590, [512,2])
  #[512,2]
  var12592=tf.add(var12591, var12451)
  #[512,1,2]
  var12593=tf.reshape(var12592, [512,1,2])
  #[512,32]
  var12594=tf.multiply(var12369, var12587)
  #[512,1,32]
  var12595=tf.reshape(var12594, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12596=var12426[:,8:9]
  #[512]
  var12597=tf.reshape(var12596, [512])
  #[512,150]
  var12598=tf.gather(params=var12425, indices=var12597, batch_dims=0, axis=0)
  #[512,150]
  var12599=tf.multiply(var12424, var12598)
  #[512,1024]
  var12600=tf.gather(params=var12599, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12601=tf.where(var12414, var12600, var12437)
  #[512,32,32]
  var12602=tf.reshape(var12601, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12603=tf.transpose(var12602, perm=[0,2,1])
  #[512,32,32]
  var12604=tf.subtract(var12602, var12603)
  #[512,32,32]
  var12605=tf.linalg.expm(var12604)
  #[512,1,32]
  var12606=tf.matmul(var12595, var12605)
  #[512,32]
  var12607=tf.reshape(var12606, [512,32])
  #[512,32]
  var12608=tf.multiply(var12365, var12607)
  #[512,32]
  var12609=tf.reshape(var12608, [512,32])
  #[512,2]
  var12610=tf.matmul(var12609, var12447)
  #[512,2]
  var12611=tf.reshape(var12610, [512,2])
  #[512,2]
  var12612=tf.add(var12611, var12451)
  #[512,1,2]
  var12613=tf.reshape(var12612, [512,1,2])
  #[512,32]
  var12614=tf.multiply(var12369, var12607)
  #[512,1,32]
  var12615=tf.reshape(var12614, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12616=var12426[:,9:10]
  #[512]
  var12617=tf.reshape(var12616, [512])
  #[512,150]
  var12618=tf.gather(params=var12425, indices=var12617, batch_dims=0, axis=0)
  #[512,150]
  var12619=tf.multiply(var12424, var12618)
  #[512,1024]
  var12620=tf.gather(params=var12619, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12621=tf.where(var12414, var12620, var12437)
  #[512,32,32]
  var12622=tf.reshape(var12621, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12623=tf.transpose(var12622, perm=[0,2,1])
  #[512,32,32]
  var12624=tf.subtract(var12622, var12623)
  #[512,32,32]
  var12625=tf.linalg.expm(var12624)
  #[512,1,32]
  var12626=tf.matmul(var12615, var12625)
  #[512,32]
  var12627=tf.reshape(var12626, [512,32])
  #[512,32]
  var12628=tf.multiply(var12365, var12627)
  #[512,32]
  var12629=tf.reshape(var12628, [512,32])
  #[512,2]
  var12630=tf.matmul(var12629, var12447)
  #[512,2]
  var12631=tf.reshape(var12630, [512,2])
  #[512,2]
  var12632=tf.add(var12631, var12451)
  #[512,1,2]
  var12633=tf.reshape(var12632, [512,1,2])
  #[512,32]
  var12634=tf.multiply(var12369, var12627)
  #[512,1,32]
  var12635=tf.reshape(var12634, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12636=var12426[:,10:11]
  #[512]
  var12637=tf.reshape(var12636, [512])
  #[512,150]
  var12638=tf.gather(params=var12425, indices=var12637, batch_dims=0, axis=0)
  #[512,150]
  var12639=tf.multiply(var12424, var12638)
  #[512,1024]
  var12640=tf.gather(params=var12639, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12641=tf.where(var12414, var12640, var12437)
  #[512,32,32]
  var12642=tf.reshape(var12641, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12643=tf.transpose(var12642, perm=[0,2,1])
  #[512,32,32]
  var12644=tf.subtract(var12642, var12643)
  #[512,32,32]
  var12645=tf.linalg.expm(var12644)
  #[512,1,32]
  var12646=tf.matmul(var12635, var12645)
  #[512,32]
  var12647=tf.reshape(var12646, [512,32])
  #[512,32]
  var12648=tf.multiply(var12365, var12647)
  #[512,32]
  var12649=tf.reshape(var12648, [512,32])
  #[512,2]
  var12650=tf.matmul(var12649, var12447)
  #[512,2]
  var12651=tf.reshape(var12650, [512,2])
  #[512,2]
  var12652=tf.add(var12651, var12451)
  #[512,1,2]
  var12653=tf.reshape(var12652, [512,1,2])
  #[512,32]
  var12654=tf.multiply(var12369, var12647)
  #[512,1,32]
  var12655=tf.reshape(var12654, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12656=var12426[:,11:12]
  #[512]
  var12657=tf.reshape(var12656, [512])
  #[512,150]
  var12658=tf.gather(params=var12425, indices=var12657, batch_dims=0, axis=0)
  #[512,150]
  var12659=tf.multiply(var12424, var12658)
  #[512,1024]
  var12660=tf.gather(params=var12659, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12661=tf.where(var12414, var12660, var12437)
  #[512,32,32]
  var12662=tf.reshape(var12661, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12663=tf.transpose(var12662, perm=[0,2,1])
  #[512,32,32]
  var12664=tf.subtract(var12662, var12663)
  #[512,32,32]
  var12665=tf.linalg.expm(var12664)
  #[512,1,32]
  var12666=tf.matmul(var12655, var12665)
  #[512,32]
  var12667=tf.reshape(var12666, [512,32])
  #[512,32]
  var12668=tf.multiply(var12365, var12667)
  #[512,32]
  var12669=tf.reshape(var12668, [512,32])
  #[512,2]
  var12670=tf.matmul(var12669, var12447)
  #[512,2]
  var12671=tf.reshape(var12670, [512,2])
  #[512,2]
  var12672=tf.add(var12671, var12451)
  #[512,1,2]
  var12673=tf.reshape(var12672, [512,1,2])
  #[512,32]
  var12674=tf.multiply(var12369, var12667)
  #[512,1,32]
  var12675=tf.reshape(var12674, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12676=var12426[:,12:13]
  #[512]
  var12677=tf.reshape(var12676, [512])
  #[512,150]
  var12678=tf.gather(params=var12425, indices=var12677, batch_dims=0, axis=0)
  #[512,150]
  var12679=tf.multiply(var12424, var12678)
  #[512,1024]
  var12680=tf.gather(params=var12679, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12681=tf.where(var12414, var12680, var12437)
  #[512,32,32]
  var12682=tf.reshape(var12681, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12683=tf.transpose(var12682, perm=[0,2,1])
  #[512,32,32]
  var12684=tf.subtract(var12682, var12683)
  #[512,32,32]
  var12685=tf.linalg.expm(var12684)
  #[512,1,32]
  var12686=tf.matmul(var12675, var12685)
  #[512,32]
  var12687=tf.reshape(var12686, [512,32])
  #[512,32]
  var12688=tf.multiply(var12365, var12687)
  #[512,32]
  var12689=tf.reshape(var12688, [512,32])
  #[512,2]
  var12690=tf.matmul(var12689, var12447)
  #[512,2]
  var12691=tf.reshape(var12690, [512,2])
  #[512,2]
  var12692=tf.add(var12691, var12451)
  #[512,1,2]
  var12693=tf.reshape(var12692, [512,1,2])
  #[512,32]
  var12694=tf.multiply(var12369, var12687)
  #[512,1,32]
  var12695=tf.reshape(var12694, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12696=var12426[:,13:14]
  #[512]
  var12697=tf.reshape(var12696, [512])
  #[512,150]
  var12698=tf.gather(params=var12425, indices=var12697, batch_dims=0, axis=0)
  #[512,150]
  var12699=tf.multiply(var12424, var12698)
  #[512,1024]
  var12700=tf.gather(params=var12699, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12701=tf.where(var12414, var12700, var12437)
  #[512,32,32]
  var12702=tf.reshape(var12701, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12703=tf.transpose(var12702, perm=[0,2,1])
  #[512,32,32]
  var12704=tf.subtract(var12702, var12703)
  #[512,32,32]
  var12705=tf.linalg.expm(var12704)
  #[512,1,32]
  var12706=tf.matmul(var12695, var12705)
  #[512,32]
  var12707=tf.reshape(var12706, [512,32])
  #[512,32]
  var12708=tf.multiply(var12365, var12707)
  #[512,32]
  var12709=tf.reshape(var12708, [512,32])
  #[512,2]
  var12710=tf.matmul(var12709, var12447)
  #[512,2]
  var12711=tf.reshape(var12710, [512,2])
  #[512,2]
  var12712=tf.add(var12711, var12451)
  #[512,1,2]
  var12713=tf.reshape(var12712, [512,1,2])
  #[512,32]
  var12714=tf.multiply(var12369, var12707)
  #[512,1,32]
  var12715=tf.reshape(var12714, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12716=var12426[:,14:15]
  #[512]
  var12717=tf.reshape(var12716, [512])
  #[512,150]
  var12718=tf.gather(params=var12425, indices=var12717, batch_dims=0, axis=0)
  #[512,150]
  var12719=tf.multiply(var12424, var12718)
  #[512,1024]
  var12720=tf.gather(params=var12719, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12721=tf.where(var12414, var12720, var12437)
  #[512,32,32]
  var12722=tf.reshape(var12721, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12723=tf.transpose(var12722, perm=[0,2,1])
  #[512,32,32]
  var12724=tf.subtract(var12722, var12723)
  #[512,32,32]
  var12725=tf.linalg.expm(var12724)
  #[512,1,32]
  var12726=tf.matmul(var12715, var12725)
  #[512,32]
  var12727=tf.reshape(var12726, [512,32])
  #[512,32]
  var12728=tf.multiply(var12365, var12727)
  #[512,32]
  var12729=tf.reshape(var12728, [512,32])
  #[512,2]
  var12730=tf.matmul(var12729, var12447)
  #[512,2]
  var12731=tf.reshape(var12730, [512,2])
  #[512,2]
  var12732=tf.add(var12731, var12451)
  #[512,1,2]
  var12733=tf.reshape(var12732, [512,1,2])
  #[512,32]
  var12734=tf.multiply(var12369, var12727)
  #[512,1,32]
  var12735=tf.reshape(var12734, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12736=var12426[:,15:16]
  #[512]
  var12737=tf.reshape(var12736, [512])
  #[512,150]
  var12738=tf.gather(params=var12425, indices=var12737, batch_dims=0, axis=0)
  #[512,150]
  var12739=tf.multiply(var12424, var12738)
  #[512,1024]
  var12740=tf.gather(params=var12739, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12741=tf.where(var12414, var12740, var12437)
  #[512,32,32]
  var12742=tf.reshape(var12741, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12743=tf.transpose(var12742, perm=[0,2,1])
  #[512,32,32]
  var12744=tf.subtract(var12742, var12743)
  #[512,32,32]
  var12745=tf.linalg.expm(var12744)
  #[512,1,32]
  var12746=tf.matmul(var12735, var12745)
  #[512,32]
  var12747=tf.reshape(var12746, [512,32])
  #[512,32]
  var12748=tf.multiply(var12365, var12747)
  #[512,32]
  var12749=tf.reshape(var12748, [512,32])
  #[512,2]
  var12750=tf.matmul(var12749, var12447)
  #[512,2]
  var12751=tf.reshape(var12750, [512,2])
  #[512,2]
  var12752=tf.add(var12751, var12451)
  #[512,1,2]
  var12753=tf.reshape(var12752, [512,1,2])
  #[512,32]
  var12754=tf.multiply(var12369, var12747)
  #[512,1,32]
  var12755=tf.reshape(var12754, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12756=var12426[:,16:17]
  #[512]
  var12757=tf.reshape(var12756, [512])
  #[512,150]
  var12758=tf.gather(params=var12425, indices=var12757, batch_dims=0, axis=0)
  #[512,150]
  var12759=tf.multiply(var12424, var12758)
  #[512,1024]
  var12760=tf.gather(params=var12759, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12761=tf.where(var12414, var12760, var12437)
  #[512,32,32]
  var12762=tf.reshape(var12761, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12763=tf.transpose(var12762, perm=[0,2,1])
  #[512,32,32]
  var12764=tf.subtract(var12762, var12763)
  #[512,32,32]
  var12765=tf.linalg.expm(var12764)
  #[512,1,32]
  var12766=tf.matmul(var12755, var12765)
  #[512,32]
  var12767=tf.reshape(var12766, [512,32])
  #[512,32]
  var12768=tf.multiply(var12365, var12767)
  #[512,32]
  var12769=tf.reshape(var12768, [512,32])
  #[512,2]
  var12770=tf.matmul(var12769, var12447)
  #[512,2]
  var12771=tf.reshape(var12770, [512,2])
  #[512,2]
  var12772=tf.add(var12771, var12451)
  #[512,1,2]
  var12773=tf.reshape(var12772, [512,1,2])
  #[512,32]
  var12774=tf.multiply(var12369, var12767)
  #[512,1,32]
  var12775=tf.reshape(var12774, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12776=var12426[:,17:18]
  #[512]
  var12777=tf.reshape(var12776, [512])
  #[512,150]
  var12778=tf.gather(params=var12425, indices=var12777, batch_dims=0, axis=0)
  #[512,150]
  var12779=tf.multiply(var12424, var12778)
  #[512,1024]
  var12780=tf.gather(params=var12779, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12781=tf.where(var12414, var12780, var12437)
  #[512,32,32]
  var12782=tf.reshape(var12781, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12783=tf.transpose(var12782, perm=[0,2,1])
  #[512,32,32]
  var12784=tf.subtract(var12782, var12783)
  #[512,32,32]
  var12785=tf.linalg.expm(var12784)
  #[512,1,32]
  var12786=tf.matmul(var12775, var12785)
  #[512,32]
  var12787=tf.reshape(var12786, [512,32])
  #[512,32]
  var12788=tf.multiply(var12365, var12787)
  #[512,32]
  var12789=tf.reshape(var12788, [512,32])
  #[512,2]
  var12790=tf.matmul(var12789, var12447)
  #[512,2]
  var12791=tf.reshape(var12790, [512,2])
  #[512,2]
  var12792=tf.add(var12791, var12451)
  #[512,1,2]
  var12793=tf.reshape(var12792, [512,1,2])
  #[512,32]
  var12794=tf.multiply(var12369, var12787)
  #[512,1,32]
  var12795=tf.reshape(var12794, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12796=var12426[:,18:19]
  #[512]
  var12797=tf.reshape(var12796, [512])
  #[512,150]
  var12798=tf.gather(params=var12425, indices=var12797, batch_dims=0, axis=0)
  #[512,150]
  var12799=tf.multiply(var12424, var12798)
  #[512,1024]
  var12800=tf.gather(params=var12799, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12801=tf.where(var12414, var12800, var12437)
  #[512,32,32]
  var12802=tf.reshape(var12801, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12803=tf.transpose(var12802, perm=[0,2,1])
  #[512,32,32]
  var12804=tf.subtract(var12802, var12803)
  #[512,32,32]
  var12805=tf.linalg.expm(var12804)
  #[512,1,32]
  var12806=tf.matmul(var12795, var12805)
  #[512,32]
  var12807=tf.reshape(var12806, [512,32])
  #[512,32]
  var12808=tf.multiply(var12365, var12807)
  #[512,32]
  var12809=tf.reshape(var12808, [512,32])
  #[512,2]
  var12810=tf.matmul(var12809, var12447)
  #[512,2]
  var12811=tf.reshape(var12810, [512,2])
  #[512,2]
  var12812=tf.add(var12811, var12451)
  #[512,1,2]
  var12813=tf.reshape(var12812, [512,1,2])
  #[512,32]
  var12814=tf.multiply(var12369, var12807)
  #[512,1,32]
  var12815=tf.reshape(var12814, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12816=var12426[:,19:20]
  #[512]
  var12817=tf.reshape(var12816, [512])
  #[512,150]
  var12818=tf.gather(params=var12425, indices=var12817, batch_dims=0, axis=0)
  #[512,150]
  var12819=tf.multiply(var12424, var12818)
  #[512,1024]
  var12820=tf.gather(params=var12819, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12821=tf.where(var12414, var12820, var12437)
  #[512,32,32]
  var12822=tf.reshape(var12821, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12823=tf.transpose(var12822, perm=[0,2,1])
  #[512,32,32]
  var12824=tf.subtract(var12822, var12823)
  #[512,32,32]
  var12825=tf.linalg.expm(var12824)
  #[512,1,32]
  var12826=tf.matmul(var12815, var12825)
  #[512,32]
  var12827=tf.reshape(var12826, [512,32])
  #[512,32]
  var12828=tf.multiply(var12365, var12827)
  #[512,32]
  var12829=tf.reshape(var12828, [512,32])
  #[512,2]
  var12830=tf.matmul(var12829, var12447)
  #[512,2]
  var12831=tf.reshape(var12830, [512,2])
  #[512,2]
  var12832=tf.add(var12831, var12451)
  #[512,1,2]
  var12833=tf.reshape(var12832, [512,1,2])
  #[512,32]
  var12834=tf.multiply(var12369, var12827)
  #[512,1,32]
  var12835=tf.reshape(var12834, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12836=var12426[:,20:21]
  #[512]
  var12837=tf.reshape(var12836, [512])
  #[512,150]
  var12838=tf.gather(params=var12425, indices=var12837, batch_dims=0, axis=0)
  #[512,150]
  var12839=tf.multiply(var12424, var12838)
  #[512,1024]
  var12840=tf.gather(params=var12839, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12841=tf.where(var12414, var12840, var12437)
  #[512,32,32]
  var12842=tf.reshape(var12841, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12843=tf.transpose(var12842, perm=[0,2,1])
  #[512,32,32]
  var12844=tf.subtract(var12842, var12843)
  #[512,32,32]
  var12845=tf.linalg.expm(var12844)
  #[512,1,32]
  var12846=tf.matmul(var12835, var12845)
  #[512,32]
  var12847=tf.reshape(var12846, [512,32])
  #[512,32]
  var12848=tf.multiply(var12365, var12847)
  #[512,32]
  var12849=tf.reshape(var12848, [512,32])
  #[512,2]
  var12850=tf.matmul(var12849, var12447)
  #[512,2]
  var12851=tf.reshape(var12850, [512,2])
  #[512,2]
  var12852=tf.add(var12851, var12451)
  #[512,1,2]
  var12853=tf.reshape(var12852, [512,1,2])
  #[512,32]
  var12854=tf.multiply(var12369, var12847)
  #[512,1,32]
  var12855=tf.reshape(var12854, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12856=var12426[:,21:22]
  #[512]
  var12857=tf.reshape(var12856, [512])
  #[512,150]
  var12858=tf.gather(params=var12425, indices=var12857, batch_dims=0, axis=0)
  #[512,150]
  var12859=tf.multiply(var12424, var12858)
  #[512,1024]
  var12860=tf.gather(params=var12859, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12861=tf.where(var12414, var12860, var12437)
  #[512,32,32]
  var12862=tf.reshape(var12861, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12863=tf.transpose(var12862, perm=[0,2,1])
  #[512,32,32]
  var12864=tf.subtract(var12862, var12863)
  #[512,32,32]
  var12865=tf.linalg.expm(var12864)
  #[512,1,32]
  var12866=tf.matmul(var12855, var12865)
  #[512,32]
  var12867=tf.reshape(var12866, [512,32])
  #[512,32]
  var12868=tf.multiply(var12365, var12867)
  #[512,32]
  var12869=tf.reshape(var12868, [512,32])
  #[512,2]
  var12870=tf.matmul(var12869, var12447)
  #[512,2]
  var12871=tf.reshape(var12870, [512,2])
  #[512,2]
  var12872=tf.add(var12871, var12451)
  #[512,1,2]
  var12873=tf.reshape(var12872, [512,1,2])
  #[512,32]
  var12874=tf.multiply(var12369, var12867)
  #[512,1,32]
  var12875=tf.reshape(var12874, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12876=var12426[:,22:23]
  #[512]
  var12877=tf.reshape(var12876, [512])
  #[512,150]
  var12878=tf.gather(params=var12425, indices=var12877, batch_dims=0, axis=0)
  #[512,150]
  var12879=tf.multiply(var12424, var12878)
  #[512,1024]
  var12880=tf.gather(params=var12879, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12881=tf.where(var12414, var12880, var12437)
  #[512,32,32]
  var12882=tf.reshape(var12881, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12883=tf.transpose(var12882, perm=[0,2,1])
  #[512,32,32]
  var12884=tf.subtract(var12882, var12883)
  #[512,32,32]
  var12885=tf.linalg.expm(var12884)
  #[512,1,32]
  var12886=tf.matmul(var12875, var12885)
  #[512,32]
  var12887=tf.reshape(var12886, [512,32])
  #[512,32]
  var12888=tf.multiply(var12365, var12887)
  #[512,32]
  var12889=tf.reshape(var12888, [512,32])
  #[512,2]
  var12890=tf.matmul(var12889, var12447)
  #[512,2]
  var12891=tf.reshape(var12890, [512,2])
  #[512,2]
  var12892=tf.add(var12891, var12451)
  #[512,1,2]
  var12893=tf.reshape(var12892, [512,1,2])
  #[512,32]
  var12894=tf.multiply(var12369, var12887)
  #[512,1,32]
  var12895=tf.reshape(var12894, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12896=var12426[:,23:24]
  #[512]
  var12897=tf.reshape(var12896, [512])
  #[512,150]
  var12898=tf.gather(params=var12425, indices=var12897, batch_dims=0, axis=0)
  #[512,150]
  var12899=tf.multiply(var12424, var12898)
  #[512,1024]
  var12900=tf.gather(params=var12899, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12901=tf.where(var12414, var12900, var12437)
  #[512,32,32]
  var12902=tf.reshape(var12901, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12903=tf.transpose(var12902, perm=[0,2,1])
  #[512,32,32]
  var12904=tf.subtract(var12902, var12903)
  #[512,32,32]
  var12905=tf.linalg.expm(var12904)
  #[512,1,32]
  var12906=tf.matmul(var12895, var12905)
  #[512,32]
  var12907=tf.reshape(var12906, [512,32])
  #[512,32]
  var12908=tf.multiply(var12365, var12907)
  #[512,32]
  var12909=tf.reshape(var12908, [512,32])
  #[512,2]
  var12910=tf.matmul(var12909, var12447)
  #[512,2]
  var12911=tf.reshape(var12910, [512,2])
  #[512,2]
  var12912=tf.add(var12911, var12451)
  #[512,1,2]
  var12913=tf.reshape(var12912, [512,1,2])
  #[512,32]
  var12914=tf.multiply(var12369, var12907)
  #[512,1,32]
  var12915=tf.reshape(var12914, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12916=var12426[:,24:25]
  #[512]
  var12917=tf.reshape(var12916, [512])
  #[512,150]
  var12918=tf.gather(params=var12425, indices=var12917, batch_dims=0, axis=0)
  #[512,150]
  var12919=tf.multiply(var12424, var12918)
  #[512,1024]
  var12920=tf.gather(params=var12919, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12921=tf.where(var12414, var12920, var12437)
  #[512,32,32]
  var12922=tf.reshape(var12921, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12923=tf.transpose(var12922, perm=[0,2,1])
  #[512,32,32]
  var12924=tf.subtract(var12922, var12923)
  #[512,32,32]
  var12925=tf.linalg.expm(var12924)
  #[512,1,32]
  var12926=tf.matmul(var12915, var12925)
  #[512,32]
  var12927=tf.reshape(var12926, [512,32])
  #[512,32]
  var12928=tf.multiply(var12365, var12927)
  #[512,32]
  var12929=tf.reshape(var12928, [512,32])
  #[512,2]
  var12930=tf.matmul(var12929, var12447)
  #[512,2]
  var12931=tf.reshape(var12930, [512,2])
  #[512,2]
  var12932=tf.add(var12931, var12451)
  #[512,1,2]
  var12933=tf.reshape(var12932, [512,1,2])
  #[512,32]
  var12934=tf.multiply(var12369, var12927)
  #[512,1,32]
  var12935=tf.reshape(var12934, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12936=var12426[:,25:26]
  #[512]
  var12937=tf.reshape(var12936, [512])
  #[512,150]
  var12938=tf.gather(params=var12425, indices=var12937, batch_dims=0, axis=0)
  #[512,150]
  var12939=tf.multiply(var12424, var12938)
  #[512,1024]
  var12940=tf.gather(params=var12939, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12941=tf.where(var12414, var12940, var12437)
  #[512,32,32]
  var12942=tf.reshape(var12941, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12943=tf.transpose(var12942, perm=[0,2,1])
  #[512,32,32]
  var12944=tf.subtract(var12942, var12943)
  #[512,32,32]
  var12945=tf.linalg.expm(var12944)
  #[512,1,32]
  var12946=tf.matmul(var12935, var12945)
  #[512,32]
  var12947=tf.reshape(var12946, [512,32])
  #[512,32]
  var12948=tf.multiply(var12365, var12947)
  #[512,32]
  var12949=tf.reshape(var12948, [512,32])
  #[512,2]
  var12950=tf.matmul(var12949, var12447)
  #[512,2]
  var12951=tf.reshape(var12950, [512,2])
  #[512,2]
  var12952=tf.add(var12951, var12451)
  #[512,1,2]
  var12953=tf.reshape(var12952, [512,1,2])
  #[512,32]
  var12954=tf.multiply(var12369, var12947)
  #[512,1,32]
  var12955=tf.reshape(var12954, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12956=var12426[:,26:27]
  #[512]
  var12957=tf.reshape(var12956, [512])
  #[512,150]
  var12958=tf.gather(params=var12425, indices=var12957, batch_dims=0, axis=0)
  #[512,150]
  var12959=tf.multiply(var12424, var12958)
  #[512,1024]
  var12960=tf.gather(params=var12959, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12961=tf.where(var12414, var12960, var12437)
  #[512,32,32]
  var12962=tf.reshape(var12961, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12963=tf.transpose(var12962, perm=[0,2,1])
  #[512,32,32]
  var12964=tf.subtract(var12962, var12963)
  #[512,32,32]
  var12965=tf.linalg.expm(var12964)
  #[512,1,32]
  var12966=tf.matmul(var12955, var12965)
  #[512,32]
  var12967=tf.reshape(var12966, [512,32])
  #[512,32]
  var12968=tf.multiply(var12365, var12967)
  #[512,32]
  var12969=tf.reshape(var12968, [512,32])
  #[512,2]
  var12970=tf.matmul(var12969, var12447)
  #[512,2]
  var12971=tf.reshape(var12970, [512,2])
  #[512,2]
  var12972=tf.add(var12971, var12451)
  #[512,1,2]
  var12973=tf.reshape(var12972, [512,1,2])
  #[512,32]
  var12974=tf.multiply(var12369, var12967)
  #[512,1,32]
  var12975=tf.reshape(var12974, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12976=var12426[:,27:28]
  #[512]
  var12977=tf.reshape(var12976, [512])
  #[512,150]
  var12978=tf.gather(params=var12425, indices=var12977, batch_dims=0, axis=0)
  #[512,150]
  var12979=tf.multiply(var12424, var12978)
  #[512,1024]
  var12980=tf.gather(params=var12979, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var12981=tf.where(var12414, var12980, var12437)
  #[512,32,32]
  var12982=tf.reshape(var12981, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var12983=tf.transpose(var12982, perm=[0,2,1])
  #[512,32,32]
  var12984=tf.subtract(var12982, var12983)
  #[512,32,32]
  var12985=tf.linalg.expm(var12984)
  #[512,1,32]
  var12986=tf.matmul(var12975, var12985)
  #[512,32]
  var12987=tf.reshape(var12986, [512,32])
  #[512,32]
  var12988=tf.multiply(var12365, var12987)
  #[512,32]
  var12989=tf.reshape(var12988, [512,32])
  #[512,2]
  var12990=tf.matmul(var12989, var12447)
  #[512,2]
  var12991=tf.reshape(var12990, [512,2])
  #[512,2]
  var12992=tf.add(var12991, var12451)
  #[512,1,2]
  var12993=tf.reshape(var12992, [512,1,2])
  #[512,32]
  var12994=tf.multiply(var12369, var12987)
  #[512,1,32]
  var12995=tf.reshape(var12994, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var12996=var12426[:,28:29]
  #[512]
  var12997=tf.reshape(var12996, [512])
  #[512,150]
  var12998=tf.gather(params=var12425, indices=var12997, batch_dims=0, axis=0)
  #[512,150]
  var12999=tf.multiply(var12424, var12998)
  #[512,1024]
  var13000=tf.gather(params=var12999, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13001=tf.where(var12414, var13000, var12437)
  #[512,32,32]
  var13002=tf.reshape(var13001, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13003=tf.transpose(var13002, perm=[0,2,1])
  #[512,32,32]
  var13004=tf.subtract(var13002, var13003)
  #[512,32,32]
  var13005=tf.linalg.expm(var13004)
  #[512,1,32]
  var13006=tf.matmul(var12995, var13005)
  #[512,32]
  var13007=tf.reshape(var13006, [512,32])
  #[512,32]
  var13008=tf.multiply(var12365, var13007)
  #[512,32]
  var13009=tf.reshape(var13008, [512,32])
  #[512,2]
  var13010=tf.matmul(var13009, var12447)
  #[512,2]
  var13011=tf.reshape(var13010, [512,2])
  #[512,2]
  var13012=tf.add(var13011, var12451)
  #[512,1,2]
  var13013=tf.reshape(var13012, [512,1,2])
  #[512,32]
  var13014=tf.multiply(var12369, var13007)
  #[512,1,32]
  var13015=tf.reshape(var13014, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13016=var12426[:,29:30]
  #[512]
  var13017=tf.reshape(var13016, [512])
  #[512,150]
  var13018=tf.gather(params=var12425, indices=var13017, batch_dims=0, axis=0)
  #[512,150]
  var13019=tf.multiply(var12424, var13018)
  #[512,1024]
  var13020=tf.gather(params=var13019, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13021=tf.where(var12414, var13020, var12437)
  #[512,32,32]
  var13022=tf.reshape(var13021, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13023=tf.transpose(var13022, perm=[0,2,1])
  #[512,32,32]
  var13024=tf.subtract(var13022, var13023)
  #[512,32,32]
  var13025=tf.linalg.expm(var13024)
  #[512,1,32]
  var13026=tf.matmul(var13015, var13025)
  #[512,32]
  var13027=tf.reshape(var13026, [512,32])
  #[512,32]
  var13028=tf.multiply(var12365, var13027)
  #[512,32]
  var13029=tf.reshape(var13028, [512,32])
  #[512,2]
  var13030=tf.matmul(var13029, var12447)
  #[512,2]
  var13031=tf.reshape(var13030, [512,2])
  #[512,2]
  var13032=tf.add(var13031, var12451)
  #[512,1,2]
  var13033=tf.reshape(var13032, [512,1,2])
  #[512,32]
  var13034=tf.multiply(var12369, var13027)
  #[512,1,32]
  var13035=tf.reshape(var13034, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13036=var12426[:,30:31]
  #[512]
  var13037=tf.reshape(var13036, [512])
  #[512,150]
  var13038=tf.gather(params=var12425, indices=var13037, batch_dims=0, axis=0)
  #[512,150]
  var13039=tf.multiply(var12424, var13038)
  #[512,1024]
  var13040=tf.gather(params=var13039, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13041=tf.where(var12414, var13040, var12437)
  #[512,32,32]
  var13042=tf.reshape(var13041, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13043=tf.transpose(var13042, perm=[0,2,1])
  #[512,32,32]
  var13044=tf.subtract(var13042, var13043)
  #[512,32,32]
  var13045=tf.linalg.expm(var13044)
  #[512,1,32]
  var13046=tf.matmul(var13035, var13045)
  #[512,32]
  var13047=tf.reshape(var13046, [512,32])
  #[512,32]
  var13048=tf.multiply(var12365, var13047)
  #[512,32]
  var13049=tf.reshape(var13048, [512,32])
  #[512,2]
  var13050=tf.matmul(var13049, var12447)
  #[512,2]
  var13051=tf.reshape(var13050, [512,2])
  #[512,2]
  var13052=tf.add(var13051, var12451)
  #[512,1,2]
  var13053=tf.reshape(var13052, [512,1,2])
  #[512,32]
  var13054=tf.multiply(var12369, var13047)
  #[512,1,32]
  var13055=tf.reshape(var13054, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13056=var12426[:,31:32]
  #[512]
  var13057=tf.reshape(var13056, [512])
  #[512,150]
  var13058=tf.gather(params=var12425, indices=var13057, batch_dims=0, axis=0)
  #[512,150]
  var13059=tf.multiply(var12424, var13058)
  #[512,1024]
  var13060=tf.gather(params=var13059, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13061=tf.where(var12414, var13060, var12437)
  #[512,32,32]
  var13062=tf.reshape(var13061, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13063=tf.transpose(var13062, perm=[0,2,1])
  #[512,32,32]
  var13064=tf.subtract(var13062, var13063)
  #[512,32,32]
  var13065=tf.linalg.expm(var13064)
  #[512,1,32]
  var13066=tf.matmul(var13055, var13065)
  #[512,32]
  var13067=tf.reshape(var13066, [512,32])
  #[512,32]
  var13068=tf.multiply(var12365, var13067)
  #[512,32]
  var13069=tf.reshape(var13068, [512,32])
  #[512,2]
  var13070=tf.matmul(var13069, var12447)
  #[512,2]
  var13071=tf.reshape(var13070, [512,2])
  #[512,2]
  var13072=tf.add(var13071, var12451)
  #[512,1,2]
  var13073=tf.reshape(var13072, [512,1,2])
  #[512,32]
  var13074=tf.multiply(var12369, var13067)
  #[512,1,32]
  var13075=tf.reshape(var13074, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13076=var12426[:,32:33]
  #[512]
  var13077=tf.reshape(var13076, [512])
  #[512,150]
  var13078=tf.gather(params=var12425, indices=var13077, batch_dims=0, axis=0)
  #[512,150]
  var13079=tf.multiply(var12424, var13078)
  #[512,1024]
  var13080=tf.gather(params=var13079, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13081=tf.where(var12414, var13080, var12437)
  #[512,32,32]
  var13082=tf.reshape(var13081, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13083=tf.transpose(var13082, perm=[0,2,1])
  #[512,32,32]
  var13084=tf.subtract(var13082, var13083)
  #[512,32,32]
  var13085=tf.linalg.expm(var13084)
  #[512,1,32]
  var13086=tf.matmul(var13075, var13085)
  #[512,32]
  var13087=tf.reshape(var13086, [512,32])
  #[512,32]
  var13088=tf.multiply(var12365, var13087)
  #[512,32]
  var13089=tf.reshape(var13088, [512,32])
  #[512,2]
  var13090=tf.matmul(var13089, var12447)
  #[512,2]
  var13091=tf.reshape(var13090, [512,2])
  #[512,2]
  var13092=tf.add(var13091, var12451)
  #[512,1,2]
  var13093=tf.reshape(var13092, [512,1,2])
  #[512,32]
  var13094=tf.multiply(var12369, var13087)
  #[512,1,32]
  var13095=tf.reshape(var13094, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13096=var12426[:,33:34]
  #[512]
  var13097=tf.reshape(var13096, [512])
  #[512,150]
  var13098=tf.gather(params=var12425, indices=var13097, batch_dims=0, axis=0)
  #[512,150]
  var13099=tf.multiply(var12424, var13098)
  #[512,1024]
  var13100=tf.gather(params=var13099, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13101=tf.where(var12414, var13100, var12437)
  #[512,32,32]
  var13102=tf.reshape(var13101, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13103=tf.transpose(var13102, perm=[0,2,1])
  #[512,32,32]
  var13104=tf.subtract(var13102, var13103)
  #[512,32,32]
  var13105=tf.linalg.expm(var13104)
  #[512,1,32]
  var13106=tf.matmul(var13095, var13105)
  #[512,32]
  var13107=tf.reshape(var13106, [512,32])
  #[512,32]
  var13108=tf.multiply(var12365, var13107)
  #[512,32]
  var13109=tf.reshape(var13108, [512,32])
  #[512,2]
  var13110=tf.matmul(var13109, var12447)
  #[512,2]
  var13111=tf.reshape(var13110, [512,2])
  #[512,2]
  var13112=tf.add(var13111, var12451)
  #[512,1,2]
  var13113=tf.reshape(var13112, [512,1,2])
  #[512,32]
  var13114=tf.multiply(var12369, var13107)
  #[512,1,32]
  var13115=tf.reshape(var13114, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13116=var12426[:,34:35]
  #[512]
  var13117=tf.reshape(var13116, [512])
  #[512,150]
  var13118=tf.gather(params=var12425, indices=var13117, batch_dims=0, axis=0)
  #[512,150]
  var13119=tf.multiply(var12424, var13118)
  #[512,1024]
  var13120=tf.gather(params=var13119, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13121=tf.where(var12414, var13120, var12437)
  #[512,32,32]
  var13122=tf.reshape(var13121, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13123=tf.transpose(var13122, perm=[0,2,1])
  #[512,32,32]
  var13124=tf.subtract(var13122, var13123)
  #[512,32,32]
  var13125=tf.linalg.expm(var13124)
  #[512,1,32]
  var13126=tf.matmul(var13115, var13125)
  #[512,32]
  var13127=tf.reshape(var13126, [512,32])
  #[512,32]
  var13128=tf.multiply(var12365, var13127)
  #[512,32]
  var13129=tf.reshape(var13128, [512,32])
  #[512,2]
  var13130=tf.matmul(var13129, var12447)
  #[512,2]
  var13131=tf.reshape(var13130, [512,2])
  #[512,2]
  var13132=tf.add(var13131, var12451)
  #[512,1,2]
  var13133=tf.reshape(var13132, [512,1,2])
  #[512,32]
  var13134=tf.multiply(var12369, var13127)
  #[512,1,32]
  var13135=tf.reshape(var13134, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13136=var12426[:,35:36]
  #[512]
  var13137=tf.reshape(var13136, [512])
  #[512,150]
  var13138=tf.gather(params=var12425, indices=var13137, batch_dims=0, axis=0)
  #[512,150]
  var13139=tf.multiply(var12424, var13138)
  #[512,1024]
  var13140=tf.gather(params=var13139, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13141=tf.where(var12414, var13140, var12437)
  #[512,32,32]
  var13142=tf.reshape(var13141, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13143=tf.transpose(var13142, perm=[0,2,1])
  #[512,32,32]
  var13144=tf.subtract(var13142, var13143)
  #[512,32,32]
  var13145=tf.linalg.expm(var13144)
  #[512,1,32]
  var13146=tf.matmul(var13135, var13145)
  #[512,32]
  var13147=tf.reshape(var13146, [512,32])
  #[512,32]
  var13148=tf.multiply(var12365, var13147)
  #[512,32]
  var13149=tf.reshape(var13148, [512,32])
  #[512,2]
  var13150=tf.matmul(var13149, var12447)
  #[512,2]
  var13151=tf.reshape(var13150, [512,2])
  #[512,2]
  var13152=tf.add(var13151, var12451)
  #[512,1,2]
  var13153=tf.reshape(var13152, [512,1,2])
  #[512,32]
  var13154=tf.multiply(var12369, var13147)
  #[512,1,32]
  var13155=tf.reshape(var13154, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13156=var12426[:,36:37]
  #[512]
  var13157=tf.reshape(var13156, [512])
  #[512,150]
  var13158=tf.gather(params=var12425, indices=var13157, batch_dims=0, axis=0)
  #[512,150]
  var13159=tf.multiply(var12424, var13158)
  #[512,1024]
  var13160=tf.gather(params=var13159, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13161=tf.where(var12414, var13160, var12437)
  #[512,32,32]
  var13162=tf.reshape(var13161, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13163=tf.transpose(var13162, perm=[0,2,1])
  #[512,32,32]
  var13164=tf.subtract(var13162, var13163)
  #[512,32,32]
  var13165=tf.linalg.expm(var13164)
  #[512,1,32]
  var13166=tf.matmul(var13155, var13165)
  #[512,32]
  var13167=tf.reshape(var13166, [512,32])
  #[512,32]
  var13168=tf.multiply(var12365, var13167)
  #[512,32]
  var13169=tf.reshape(var13168, [512,32])
  #[512,2]
  var13170=tf.matmul(var13169, var12447)
  #[512,2]
  var13171=tf.reshape(var13170, [512,2])
  #[512,2]
  var13172=tf.add(var13171, var12451)
  #[512,1,2]
  var13173=tf.reshape(var13172, [512,1,2])
  #[512,32]
  var13174=tf.multiply(var12369, var13167)
  #[512,1,32]
  var13175=tf.reshape(var13174, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13176=var12426[:,37:38]
  #[512]
  var13177=tf.reshape(var13176, [512])
  #[512,150]
  var13178=tf.gather(params=var12425, indices=var13177, batch_dims=0, axis=0)
  #[512,150]
  var13179=tf.multiply(var12424, var13178)
  #[512,1024]
  var13180=tf.gather(params=var13179, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13181=tf.where(var12414, var13180, var12437)
  #[512,32,32]
  var13182=tf.reshape(var13181, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13183=tf.transpose(var13182, perm=[0,2,1])
  #[512,32,32]
  var13184=tf.subtract(var13182, var13183)
  #[512,32,32]
  var13185=tf.linalg.expm(var13184)
  #[512,1,32]
  var13186=tf.matmul(var13175, var13185)
  #[512,32]
  var13187=tf.reshape(var13186, [512,32])
  #[512,32]
  var13188=tf.multiply(var12365, var13187)
  #[512,32]
  var13189=tf.reshape(var13188, [512,32])
  #[512,2]
  var13190=tf.matmul(var13189, var12447)
  #[512,2]
  var13191=tf.reshape(var13190, [512,2])
  #[512,2]
  var13192=tf.add(var13191, var12451)
  #[512,1,2]
  var13193=tf.reshape(var13192, [512,1,2])
  #[512,32]
  var13194=tf.multiply(var12369, var13187)
  #[512,1,32]
  var13195=tf.reshape(var13194, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13196=var12426[:,38:39]
  #[512]
  var13197=tf.reshape(var13196, [512])
  #[512,150]
  var13198=tf.gather(params=var12425, indices=var13197, batch_dims=0, axis=0)
  #[512,150]
  var13199=tf.multiply(var12424, var13198)
  #[512,1024]
  var13200=tf.gather(params=var13199, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13201=tf.where(var12414, var13200, var12437)
  #[512,32,32]
  var13202=tf.reshape(var13201, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13203=tf.transpose(var13202, perm=[0,2,1])
  #[512,32,32]
  var13204=tf.subtract(var13202, var13203)
  #[512,32,32]
  var13205=tf.linalg.expm(var13204)
  #[512,1,32]
  var13206=tf.matmul(var13195, var13205)
  #[512,32]
  var13207=tf.reshape(var13206, [512,32])
  #[512,32]
  var13208=tf.multiply(var12365, var13207)
  #[512,32]
  var13209=tf.reshape(var13208, [512,32])
  #[512,2]
  var13210=tf.matmul(var13209, var12447)
  #[512,2]
  var13211=tf.reshape(var13210, [512,2])
  #[512,2]
  var13212=tf.add(var13211, var12451)
  #[512,1,2]
  var13213=tf.reshape(var13212, [512,1,2])
  #[512,32]
  var13214=tf.multiply(var12369, var13207)
  #[512,1,32]
  var13215=tf.reshape(var13214, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13216=var12426[:,39:40]
  #[512]
  var13217=tf.reshape(var13216, [512])
  #[512,150]
  var13218=tf.gather(params=var12425, indices=var13217, batch_dims=0, axis=0)
  #[512,150]
  var13219=tf.multiply(var12424, var13218)
  #[512,1024]
  var13220=tf.gather(params=var13219, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13221=tf.where(var12414, var13220, var12437)
  #[512,32,32]
  var13222=tf.reshape(var13221, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13223=tf.transpose(var13222, perm=[0,2,1])
  #[512,32,32]
  var13224=tf.subtract(var13222, var13223)
  #[512,32,32]
  var13225=tf.linalg.expm(var13224)
  #[512,1,32]
  var13226=tf.matmul(var13215, var13225)
  #[512,32]
  var13227=tf.reshape(var13226, [512,32])
  #[512,32]
  var13228=tf.multiply(var12365, var13227)
  #[512,32]
  var13229=tf.reshape(var13228, [512,32])
  #[512,2]
  var13230=tf.matmul(var13229, var12447)
  #[512,2]
  var13231=tf.reshape(var13230, [512,2])
  #[512,2]
  var13232=tf.add(var13231, var12451)
  #[512,1,2]
  var13233=tf.reshape(var13232, [512,1,2])
  #[512,32]
  var13234=tf.multiply(var12369, var13227)
  #[512,1,32]
  var13235=tf.reshape(var13234, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13236=var12426[:,40:41]
  #[512]
  var13237=tf.reshape(var13236, [512])
  #[512,150]
  var13238=tf.gather(params=var12425, indices=var13237, batch_dims=0, axis=0)
  #[512,150]
  var13239=tf.multiply(var12424, var13238)
  #[512,1024]
  var13240=tf.gather(params=var13239, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13241=tf.where(var12414, var13240, var12437)
  #[512,32,32]
  var13242=tf.reshape(var13241, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13243=tf.transpose(var13242, perm=[0,2,1])
  #[512,32,32]
  var13244=tf.subtract(var13242, var13243)
  #[512,32,32]
  var13245=tf.linalg.expm(var13244)
  #[512,1,32]
  var13246=tf.matmul(var13235, var13245)
  #[512,32]
  var13247=tf.reshape(var13246, [512,32])
  #[512,32]
  var13248=tf.multiply(var12365, var13247)
  #[512,32]
  var13249=tf.reshape(var13248, [512,32])
  #[512,2]
  var13250=tf.matmul(var13249, var12447)
  #[512,2]
  var13251=tf.reshape(var13250, [512,2])
  #[512,2]
  var13252=tf.add(var13251, var12451)
  #[512,1,2]
  var13253=tf.reshape(var13252, [512,1,2])
  #[512,32]
  var13254=tf.multiply(var12369, var13247)
  #[512,1,32]
  var13255=tf.reshape(var13254, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13256=var12426[:,41:42]
  #[512]
  var13257=tf.reshape(var13256, [512])
  #[512,150]
  var13258=tf.gather(params=var12425, indices=var13257, batch_dims=0, axis=0)
  #[512,150]
  var13259=tf.multiply(var12424, var13258)
  #[512,1024]
  var13260=tf.gather(params=var13259, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13261=tf.where(var12414, var13260, var12437)
  #[512,32,32]
  var13262=tf.reshape(var13261, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13263=tf.transpose(var13262, perm=[0,2,1])
  #[512,32,32]
  var13264=tf.subtract(var13262, var13263)
  #[512,32,32]
  var13265=tf.linalg.expm(var13264)
  #[512,1,32]
  var13266=tf.matmul(var13255, var13265)
  #[512,32]
  var13267=tf.reshape(var13266, [512,32])
  #[512,32]
  var13268=tf.multiply(var12365, var13267)
  #[512,32]
  var13269=tf.reshape(var13268, [512,32])
  #[512,2]
  var13270=tf.matmul(var13269, var12447)
  #[512,2]
  var13271=tf.reshape(var13270, [512,2])
  #[512,2]
  var13272=tf.add(var13271, var12451)
  #[512,1,2]
  var13273=tf.reshape(var13272, [512,1,2])
  #[512,32]
  var13274=tf.multiply(var12369, var13267)
  #[512,1,32]
  var13275=tf.reshape(var13274, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13276=var12426[:,42:43]
  #[512]
  var13277=tf.reshape(var13276, [512])
  #[512,150]
  var13278=tf.gather(params=var12425, indices=var13277, batch_dims=0, axis=0)
  #[512,150]
  var13279=tf.multiply(var12424, var13278)
  #[512,1024]
  var13280=tf.gather(params=var13279, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13281=tf.where(var12414, var13280, var12437)
  #[512,32,32]
  var13282=tf.reshape(var13281, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13283=tf.transpose(var13282, perm=[0,2,1])
  #[512,32,32]
  var13284=tf.subtract(var13282, var13283)
  #[512,32,32]
  var13285=tf.linalg.expm(var13284)
  #[512,1,32]
  var13286=tf.matmul(var13275, var13285)
  #[512,32]
  var13287=tf.reshape(var13286, [512,32])
  #[512,32]
  var13288=tf.multiply(var12365, var13287)
  #[512,32]
  var13289=tf.reshape(var13288, [512,32])
  #[512,2]
  var13290=tf.matmul(var13289, var12447)
  #[512,2]
  var13291=tf.reshape(var13290, [512,2])
  #[512,2]
  var13292=tf.add(var13291, var12451)
  #[512,1,2]
  var13293=tf.reshape(var13292, [512,1,2])
  #[512,32]
  var13294=tf.multiply(var12369, var13287)
  #[512,1,32]
  var13295=tf.reshape(var13294, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13296=var12426[:,43:44]
  #[512]
  var13297=tf.reshape(var13296, [512])
  #[512,150]
  var13298=tf.gather(params=var12425, indices=var13297, batch_dims=0, axis=0)
  #[512,150]
  var13299=tf.multiply(var12424, var13298)
  #[512,1024]
  var13300=tf.gather(params=var13299, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13301=tf.where(var12414, var13300, var12437)
  #[512,32,32]
  var13302=tf.reshape(var13301, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13303=tf.transpose(var13302, perm=[0,2,1])
  #[512,32,32]
  var13304=tf.subtract(var13302, var13303)
  #[512,32,32]
  var13305=tf.linalg.expm(var13304)
  #[512,1,32]
  var13306=tf.matmul(var13295, var13305)
  #[512,32]
  var13307=tf.reshape(var13306, [512,32])
  #[512,32]
  var13308=tf.multiply(var12365, var13307)
  #[512,32]
  var13309=tf.reshape(var13308, [512,32])
  #[512,2]
  var13310=tf.matmul(var13309, var12447)
  #[512,2]
  var13311=tf.reshape(var13310, [512,2])
  #[512,2]
  var13312=tf.add(var13311, var12451)
  #[512,1,2]
  var13313=tf.reshape(var13312, [512,1,2])
  #[512,32]
  var13314=tf.multiply(var12369, var13307)
  #[512,1,32]
  var13315=tf.reshape(var13314, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13316=var12426[:,44:45]
  #[512]
  var13317=tf.reshape(var13316, [512])
  #[512,150]
  var13318=tf.gather(params=var12425, indices=var13317, batch_dims=0, axis=0)
  #[512,150]
  var13319=tf.multiply(var12424, var13318)
  #[512,1024]
  var13320=tf.gather(params=var13319, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13321=tf.where(var12414, var13320, var12437)
  #[512,32,32]
  var13322=tf.reshape(var13321, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13323=tf.transpose(var13322, perm=[0,2,1])
  #[512,32,32]
  var13324=tf.subtract(var13322, var13323)
  #[512,32,32]
  var13325=tf.linalg.expm(var13324)
  #[512,1,32]
  var13326=tf.matmul(var13315, var13325)
  #[512,32]
  var13327=tf.reshape(var13326, [512,32])
  #[512,32]
  var13328=tf.multiply(var12365, var13327)
  #[512,32]
  var13329=tf.reshape(var13328, [512,32])
  #[512,2]
  var13330=tf.matmul(var13329, var12447)
  #[512,2]
  var13331=tf.reshape(var13330, [512,2])
  #[512,2]
  var13332=tf.add(var13331, var12451)
  #[512,1,2]
  var13333=tf.reshape(var13332, [512,1,2])
  #[512,32]
  var13334=tf.multiply(var12369, var13327)
  #[512,1,32]
  var13335=tf.reshape(var13334, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13336=var12426[:,45:46]
  #[512]
  var13337=tf.reshape(var13336, [512])
  #[512,150]
  var13338=tf.gather(params=var12425, indices=var13337, batch_dims=0, axis=0)
  #[512,150]
  var13339=tf.multiply(var12424, var13338)
  #[512,1024]
  var13340=tf.gather(params=var13339, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13341=tf.where(var12414, var13340, var12437)
  #[512,32,32]
  var13342=tf.reshape(var13341, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13343=tf.transpose(var13342, perm=[0,2,1])
  #[512,32,32]
  var13344=tf.subtract(var13342, var13343)
  #[512,32,32]
  var13345=tf.linalg.expm(var13344)
  #[512,1,32]
  var13346=tf.matmul(var13335, var13345)
  #[512,32]
  var13347=tf.reshape(var13346, [512,32])
  #[512,32]
  var13348=tf.multiply(var12365, var13347)
  #[512,32]
  var13349=tf.reshape(var13348, [512,32])
  #[512,2]
  var13350=tf.matmul(var13349, var12447)
  #[512,2]
  var13351=tf.reshape(var13350, [512,2])
  #[512,2]
  var13352=tf.add(var13351, var12451)
  #[512,1,2]
  var13353=tf.reshape(var13352, [512,1,2])
  #[512,32]
  var13354=tf.multiply(var12369, var13347)
  #[512,1,32]
  var13355=tf.reshape(var13354, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13356=var12426[:,46:47]
  #[512]
  var13357=tf.reshape(var13356, [512])
  #[512,150]
  var13358=tf.gather(params=var12425, indices=var13357, batch_dims=0, axis=0)
  #[512,150]
  var13359=tf.multiply(var12424, var13358)
  #[512,1024]
  var13360=tf.gather(params=var13359, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13361=tf.where(var12414, var13360, var12437)
  #[512,32,32]
  var13362=tf.reshape(var13361, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13363=tf.transpose(var13362, perm=[0,2,1])
  #[512,32,32]
  var13364=tf.subtract(var13362, var13363)
  #[512,32,32]
  var13365=tf.linalg.expm(var13364)
  #[512,1,32]
  var13366=tf.matmul(var13355, var13365)
  #[512,32]
  var13367=tf.reshape(var13366, [512,32])
  #[512,32]
  var13368=tf.multiply(var12365, var13367)
  #[512,32]
  var13369=tf.reshape(var13368, [512,32])
  #[512,2]
  var13370=tf.matmul(var13369, var12447)
  #[512,2]
  var13371=tf.reshape(var13370, [512,2])
  #[512,2]
  var13372=tf.add(var13371, var12451)
  #[512,1,2]
  var13373=tf.reshape(var13372, [512,1,2])
  #[512,32]
  var13374=tf.multiply(var12369, var13367)
  #[512,1,32]
  var13375=tf.reshape(var13374, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13376=var12426[:,47:48]
  #[512]
  var13377=tf.reshape(var13376, [512])
  #[512,150]
  var13378=tf.gather(params=var12425, indices=var13377, batch_dims=0, axis=0)
  #[512,150]
  var13379=tf.multiply(var12424, var13378)
  #[512,1024]
  var13380=tf.gather(params=var13379, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13381=tf.where(var12414, var13380, var12437)
  #[512,32,32]
  var13382=tf.reshape(var13381, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13383=tf.transpose(var13382, perm=[0,2,1])
  #[512,32,32]
  var13384=tf.subtract(var13382, var13383)
  #[512,32,32]
  var13385=tf.linalg.expm(var13384)
  #[512,1,32]
  var13386=tf.matmul(var13375, var13385)
  #[512,32]
  var13387=tf.reshape(var13386, [512,32])
  #[512,32]
  var13388=tf.multiply(var12365, var13387)
  #[512,32]
  var13389=tf.reshape(var13388, [512,32])
  #[512,2]
  var13390=tf.matmul(var13389, var12447)
  #[512,2]
  var13391=tf.reshape(var13390, [512,2])
  #[512,2]
  var13392=tf.add(var13391, var12451)
  #[512,1,2]
  var13393=tf.reshape(var13392, [512,1,2])
  #[512,32]
  var13394=tf.multiply(var12369, var13387)
  #[512,1,32]
  var13395=tf.reshape(var13394, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13396=var12426[:,48:49]
  #[512]
  var13397=tf.reshape(var13396, [512])
  #[512,150]
  var13398=tf.gather(params=var12425, indices=var13397, batch_dims=0, axis=0)
  #[512,150]
  var13399=tf.multiply(var12424, var13398)
  #[512,1024]
  var13400=tf.gather(params=var13399, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13401=tf.where(var12414, var13400, var12437)
  #[512,32,32]
  var13402=tf.reshape(var13401, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13403=tf.transpose(var13402, perm=[0,2,1])
  #[512,32,32]
  var13404=tf.subtract(var13402, var13403)
  #[512,32,32]
  var13405=tf.linalg.expm(var13404)
  #[512,1,32]
  var13406=tf.matmul(var13395, var13405)
  #[512,32]
  var13407=tf.reshape(var13406, [512,32])
  #[512,32]
  var13408=tf.multiply(var12365, var13407)
  #[512,32]
  var13409=tf.reshape(var13408, [512,32])
  #[512,2]
  var13410=tf.matmul(var13409, var12447)
  #[512,2]
  var13411=tf.reshape(var13410, [512,2])
  #[512,2]
  var13412=tf.add(var13411, var12451)
  #[512,1,2]
  var13413=tf.reshape(var13412, [512,1,2])
  #[512,32]
  var13414=tf.multiply(var12369, var13407)
  #[512,1,32]
  var13415=tf.reshape(var13414, [512,1,32])
  #transpose: p = PermSwap; [32,32]
  #[512,1]
  var13416=var12426[:,49:50]
  #[512]
  var13417=tf.reshape(var13416, [512])
  #[512,150]
  var13418=tf.gather(params=var12425, indices=var13417, batch_dims=0, axis=0)
  #[512,150]
  var13419=tf.multiply(var12424, var13418)
  #[512,1024]
  var13420=tf.gather(params=var13419, indices=var12431, batch_dims=1, axis=1)
  #[512,1024]
  var13421=tf.where(var12414, var13420, var12437)
  #[512,32,32]
  var13422=tf.reshape(var13421, [512,32,32])
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,32,32]
  var13423=tf.transpose(var13422, perm=[0,2,1])
  #[512,32,32]
  var13424=tf.subtract(var13422, var13423)
  #[512,32,32]
  var13425=tf.linalg.expm(var13424)
  #[512,1,32]
  var13426=tf.matmul(var13415, var13425)
  #[512,32]
  var13427=tf.reshape(var13426, [512,32])
  #[512,32]
  var13428=tf.multiply(var12365, var13427)
  #[512,32]
  var13429=tf.reshape(var13428, [512,32])
  #[512,2]
  var13430=tf.matmul(var13429, var12447)
  #[512,2]
  var13431=tf.reshape(var13430, [512,2])
  #[512,2]
  var13432=tf.add(var13431, var12451)
  #[512,1,2]
  var13433=tf.reshape(var13432, [512,1,2])
  #[512,50,2]
  var13434=tf.concat([var12453
                     ,var12473
                     ,var12493
                     ,var12513
                     ,var12533
                     ,var12553
                     ,var12573
                     ,var12593
                     ,var12613
                     ,var12633
                     ,var12653
                     ,var12673
                     ,var12693
                     ,var12713
                     ,var12733
                     ,var12753
                     ,var12773
                     ,var12793
                     ,var12813
                     ,var12833
                     ,var12853
                     ,var12873
                     ,var12893
                     ,var12913
                     ,var12933
                     ,var12953
                     ,var12973
                     ,var12993
                     ,var13013
                     ,var13033
                     ,var13053
                     ,var13073
                     ,var13093
                     ,var13113
                     ,var13133
                     ,var13153
                     ,var13173
                     ,var13193
                     ,var13213
                     ,var13233
                     ,var13253
                     ,var13273
                     ,var13293
                     ,var13313
                     ,var13333
                     ,var13353
                     ,var13373
                     ,var13393
                     ,var13413
                     ,var13433],
                     axis=1)
  #[512]
  var13435=yIndex
  #[512,2]
  var13436=tf.gather(params=var13434, indices=var13435, batch_dims=1, axis=1)
  #[512]
  var13437=tf.nn.softmax_cross_entropy_with_logits(labels=var12352, logits=var13436)
  #[512]
  var13438=tf.reshape(var13437, [512])
  #[]
  var13439=tf.reduce_mean(var13438, axis=0)
  #[]
  var13440=tf.add(var13439, var12435)
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
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512]
  var13441=tf.argmax(var13436, axis=1, output_type=tf.int32)
  #[512]
  var13442=tf.argmax(var12352, axis=1, output_type=tf.int32)
  #[512]
  var13443=tf.equal(var13441, var13442)
  #[512]
  var13444=tf.cast(var13443, tf.float32)
  #[512]
  var13445=tf.reshape(var13444, [512])
  #[]
  var13446=tf.reduce_mean(var13445, axis=0)
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
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #transpose: p = PermSkip PermSwap; [512,32,32]
  #[512,2]
  var13447=tf.reshape(var13436, [512,2])
  #[512,2]
  var13448=tf.nn.softmax(var13447, axis=1)
  #[512,2]
  var13449=tf.reshape(var13448, [512,2])
  return {"loss":var13440,"accuracy":var13446,"y_":var13449}
runModel = {"function":runModel_fn
           ,"batched":True
           ,"placeholders":{"x":{"shape":[512,50],"dtype":tf.int32}
                           ,"yIndex":{"shape":[512],"dtype":tf.int32}
                           ,"y":{"shape":[512],"dtype":tf.int32}}}
@tf.function
def probeEmbs_fn(training_placeholder, embs, dense_w, dense_bias, wordIdx):
  
  #[32]
  var13450=tf.range(start=0, limit=32, dtype=tf.int32)
  #[32,32]
  var13451=tf.broadcast_to(tf.reshape(var13450, [1,32]), [32,32])
  #[1024]
  var13452=tf.reshape(var13451, [1024])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13453=tf.transpose(var13451, perm=[1,0])
  #[1024]
  var13454=tf.reshape(var13453, [1024])
  #[1024]
  var13455=tf.subtract(var13452, var13454)
  #[]
  var13456=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var13457=tf.broadcast_to(tf.reshape(var13456, [1]), [1])
  #[]
  var13458=tf.reshape(var13457, [])
  #[1024]
  var13459=tf.broadcast_to(tf.reshape(var13458, [1]), [1024])
  #[1024]
  var13460=tf.math.greater(var13455, var13459)
  #transpose: p = PermSwap; [32,32]
  #[]
  var13461=tf.constant(2, shape=[], dtype=tf.int32)
  #[1]
  var13462=tf.broadcast_to(tf.reshape(var13461, [1]), [1])
  #[]
  var13463=tf.reshape(var13462, [])
  #[]
  var13464=tf.constant(32, shape=[], dtype=tf.int32)
  #[1]
  var13465=tf.broadcast_to(tf.reshape(var13464, [1]), [1])
  #[]
  var13466=tf.reshape(var13465, [])
  #[]
  var13467=tf.multiply(var13463, var13466)
  #[1024]
  var13468=tf.broadcast_to(tf.reshape(var13467, [1]), [1024])
  #[1024]
  var13469=tf.subtract(var13468, var13454)
  #[]
  var13470=tf.constant(3, shape=[], dtype=tf.int32)
  #[1]
  var13471=tf.broadcast_to(tf.reshape(var13470, [1]), [1])
  #[]
  var13472=tf.reshape(var13471, [])
  #[1024]
  var13473=tf.broadcast_to(tf.reshape(var13472, [1]), [1024])
  #[1024]
  var13474=tf.subtract(var13469, var13473)
  #[1024]
  var13475=tf.multiply(var13454, var13474)
  #[1024]
  var13476=tf.broadcast_to(tf.reshape(var13463, [1]), [1024])
  #[1024]
  var13477=tf.math.floordiv(var13475, var13476)
  #[1024]
  var13478=tf.add(var13477, var13452)
  #[]
  var13479=tf.constant(1, shape=[], dtype=tf.int32)
  #[1]
  var13480=tf.broadcast_to(tf.reshape(var13479, [1]), [1])
  #[]
  var13481=tf.reshape(var13480, [])
  #[1024]
  var13482=tf.broadcast_to(tf.reshape(var13481, [1]), [1024])
  #[1024]
  var13483=tf.subtract(var13478, var13482)
  #[]
  var13484=tf.constant(150, shape=[], dtype=tf.int32)
  #[1]
  var13485=tf.broadcast_to(tf.reshape(var13484, [1]), [1])
  #[]
  var13486=tf.reshape(var13485, [])
  #[1024]
  var13487=tf.broadcast_to(tf.reshape(var13486, [1]), [1024])
  #[1024]
  var13488=tf.math.less(var13483, var13487)
  #[1024]
  var13489=tf.math.logical_and(var13460, var13488)
  #[50050,150]
  var13490=embs
  #[]
  var13491=wordIdx
  #[150]
  var13492=tf.gather(params=var13490, indices=var13491, batch_dims=0, axis=0)
  #[1024]
  var13493=tf.gather(params=var13492, indices=var13483, batch_dims=0, axis=0)
  #[]
  var13494=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var13495=tf.broadcast_to(tf.reshape(var13494, [1]), [1])
  #[]
  var13496=tf.reshape(var13495, [])
  #[1024]
  var13497=tf.broadcast_to(tf.reshape(var13496, [1]), [1024])
  #[1024]
  var13498=tf.where(var13489, var13493, var13497)
  #[32,32]
  var13499=tf.reshape(var13498, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13500=tf.transpose(var13499, perm=[1,0])
  #[32,32]
  var13501=tf.subtract(var13499, var13500)
  return {"embsAntiHermitian":var13501}
probeEmbs = {"function":probeEmbs_fn
            ,"batched":False
            ,"placeholders":{"wordIdx":{"shape":[],"dtype":tf.int32}}}
@tf.function
def probePreds_fn(training_placeholder, embs, dense_w, dense_bias, x):
  
  #[]
  var13502=training_placeholder
  #[32]
  var13503=tf.random.uniform([32], minval=0.95, maxval=1.95, dtype=tf.float32) # 2
  #[32]
  var13504=tf.floor(var13503)
  #[]
  var13505=tf.constant(0.95, shape=[], dtype=tf.float32)
  #[32]
  var13506=tf.broadcast_to(tf.reshape(var13505, [1]), [32])
  #[32]
  var13507=tf.reshape(var13506, [32])
  #[32]
  var13508=tf.divide(var13504, var13507)
  #[]
  var13509=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[32]
  var13510=tf.broadcast_to(tf.reshape(var13509, [1]), [32])
  #[32]
  var13511=tf.reshape(var13510, [32])
  #[32]
  var13512=tf.cond(var13502, true_fn=lambda: var13508, false_fn=lambda: var13511)
  #[32]
  var13513=tf.random.uniform([32], minval=0.95, maxval=1.95, dtype=tf.float32) # 1
  #[32]
  var13514=tf.floor(var13513)
  #[32]
  var13515=tf.divide(var13514, var13507)
  #[32]
  var13516=tf.cond(var13502, true_fn=lambda: var13515, false_fn=lambda: var13511)
  #[]
  var13517=tf.constant(0, shape=[], dtype=tf.int32)
  #[1]
  var13518=tf.broadcast_to(tf.reshape(var13517, [1]), [1])
  #[]
  var13519=tf.reshape(var13518, [])
  #[32]
  var13520=tf.one_hot(var13519, axis=0, dtype=tf.float32, depth=32)
  #[32]
  var13521=tf.multiply(var13516, var13520)
  #[1,32]
  var13522=tf.reshape(var13521, [1,32])
  #[32]
  var13523=tf.range(start=0, limit=32, dtype=tf.int32)
  #[32,32]
  var13524=tf.broadcast_to(tf.reshape(var13523, [1,32]), [32,32])
  #[1024]
  var13525=tf.reshape(var13524, [1024])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13526=tf.transpose(var13524, perm=[1,0])
  #[1024]
  var13527=tf.reshape(var13526, [1024])
  #[1024]
  var13528=tf.subtract(var13525, var13527)
  #[1024]
  var13529=tf.broadcast_to(tf.reshape(var13519, [1]), [1024])
  #[1024]
  var13530=tf.math.greater(var13528, var13529)
  #transpose: p = PermSwap; [32,32]
  #[]
  var13531=tf.constant(2, shape=[], dtype=tf.int32)
  #[1]
  var13532=tf.broadcast_to(tf.reshape(var13531, [1]), [1])
  #[]
  var13533=tf.reshape(var13532, [])
  #[]
  var13534=tf.constant(32, shape=[], dtype=tf.int32)
  #[1]
  var13535=tf.broadcast_to(tf.reshape(var13534, [1]), [1])
  #[]
  var13536=tf.reshape(var13535, [])
  #[]
  var13537=tf.multiply(var13533, var13536)
  #[1024]
  var13538=tf.broadcast_to(tf.reshape(var13537, [1]), [1024])
  #[1024]
  var13539=tf.subtract(var13538, var13527)
  #[]
  var13540=tf.constant(3, shape=[], dtype=tf.int32)
  #[1]
  var13541=tf.broadcast_to(tf.reshape(var13540, [1]), [1])
  #[]
  var13542=tf.reshape(var13541, [])
  #[1024]
  var13543=tf.broadcast_to(tf.reshape(var13542, [1]), [1024])
  #[1024]
  var13544=tf.subtract(var13539, var13543)
  #[1024]
  var13545=tf.multiply(var13527, var13544)
  #[1024]
  var13546=tf.broadcast_to(tf.reshape(var13533, [1]), [1024])
  #[1024]
  var13547=tf.math.floordiv(var13545, var13546)
  #[1024]
  var13548=tf.add(var13547, var13525)
  #[]
  var13549=tf.constant(1, shape=[], dtype=tf.int32)
  #[1]
  var13550=tf.broadcast_to(tf.reshape(var13549, [1]), [1])
  #[]
  var13551=tf.reshape(var13550, [])
  #[1024]
  var13552=tf.broadcast_to(tf.reshape(var13551, [1]), [1024])
  #[1024]
  var13553=tf.subtract(var13548, var13552)
  #[]
  var13554=tf.constant(150, shape=[], dtype=tf.int32)
  #[1]
  var13555=tf.broadcast_to(tf.reshape(var13554, [1]), [1])
  #[]
  var13556=tf.reshape(var13555, [])
  #[1024]
  var13557=tf.broadcast_to(tf.reshape(var13556, [1]), [1024])
  #[1024]
  var13558=tf.math.less(var13553, var13557)
  #[1024]
  var13559=tf.math.logical_and(var13530, var13558)
  #[150]
  var13560=tf.random.uniform([150], minval=0.95, maxval=1.95, dtype=tf.float32) # 3
  #[150]
  var13561=tf.floor(var13560)
  #[150]
  var13562=tf.broadcast_to(tf.reshape(var13505, [1]), [150])
  #[150]
  var13563=tf.reshape(var13562, [150])
  #[150]
  var13564=tf.divide(var13561, var13563)
  #[150]
  var13565=tf.broadcast_to(tf.reshape(var13509, [1]), [150])
  #[150]
  var13566=tf.reshape(var13565, [150])
  #[150]
  var13567=tf.cond(var13502, true_fn=lambda: var13564, false_fn=lambda: var13566)
  #[50050,150]
  var13568=embs
  #[50]
  var13569=x
  #[1]
  var13570=var13569[0:1]
  #[]
  var13571=tf.reshape(var13570, [])
  #[150]
  var13572=tf.gather(params=var13568, indices=var13571, batch_dims=0, axis=0)
  #[150]
  var13573=tf.multiply(var13567, var13572)
  #[1024]
  var13574=tf.gather(params=var13573, indices=var13553, batch_dims=0, axis=0)
  #[]
  var13575=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var13576=tf.broadcast_to(tf.reshape(var13575, [1]), [1])
  #[]
  var13577=tf.reshape(var13576, [])
  #[1024]
  var13578=tf.broadcast_to(tf.reshape(var13577, [1]), [1024])
  #[1024]
  var13579=tf.where(var13559, var13574, var13578)
  #[32,32]
  var13580=tf.reshape(var13579, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13581=tf.transpose(var13580, perm=[1,0])
  #[32,32]
  var13582=tf.subtract(var13580, var13581)
  #[32,32]
  var13583=tf.linalg.expm(var13582)
  #[1,32]
  var13584=tf.matmul(var13522, var13583)
  #[32]
  var13585=tf.reshape(var13584, [32])
  #[32]
  var13586=tf.multiply(var13512, var13585)
  #[1,32]
  var13587=tf.reshape(var13586, [1,32])
  #[32,2]
  var13588=dense_w
  #[1,2]
  var13589=tf.matmul(var13587, var13588)
  #[2]
  var13590=tf.reshape(var13589, [2])
  #[2]
  var13591=dense_bias
  #[2]
  var13592=tf.add(var13590, var13591)
  #[1,2]
  var13593=tf.reshape(var13592, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13594=tf.multiply(var13516, var13585)
  #[1,32]
  var13595=tf.reshape(var13594, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13596=var13569[1:2]
  #[]
  var13597=tf.reshape(var13596, [])
  #[150]
  var13598=tf.gather(params=var13568, indices=var13597, batch_dims=0, axis=0)
  #[150]
  var13599=tf.multiply(var13567, var13598)
  #[1024]
  var13600=tf.gather(params=var13599, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13601=tf.where(var13559, var13600, var13578)
  #[32,32]
  var13602=tf.reshape(var13601, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13603=tf.transpose(var13602, perm=[1,0])
  #[32,32]
  var13604=tf.subtract(var13602, var13603)
  #[32,32]
  var13605=tf.linalg.expm(var13604)
  #[1,32]
  var13606=tf.matmul(var13595, var13605)
  #[32]
  var13607=tf.reshape(var13606, [32])
  #[32]
  var13608=tf.multiply(var13512, var13607)
  #[1,32]
  var13609=tf.reshape(var13608, [1,32])
  #[1,2]
  var13610=tf.matmul(var13609, var13588)
  #[2]
  var13611=tf.reshape(var13610, [2])
  #[2]
  var13612=tf.add(var13611, var13591)
  #[1,2]
  var13613=tf.reshape(var13612, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13614=tf.multiply(var13516, var13607)
  #[1,32]
  var13615=tf.reshape(var13614, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13616=var13569[2:3]
  #[]
  var13617=tf.reshape(var13616, [])
  #[150]
  var13618=tf.gather(params=var13568, indices=var13617, batch_dims=0, axis=0)
  #[150]
  var13619=tf.multiply(var13567, var13618)
  #[1024]
  var13620=tf.gather(params=var13619, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13621=tf.where(var13559, var13620, var13578)
  #[32,32]
  var13622=tf.reshape(var13621, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13623=tf.transpose(var13622, perm=[1,0])
  #[32,32]
  var13624=tf.subtract(var13622, var13623)
  #[32,32]
  var13625=tf.linalg.expm(var13624)
  #[1,32]
  var13626=tf.matmul(var13615, var13625)
  #[32]
  var13627=tf.reshape(var13626, [32])
  #[32]
  var13628=tf.multiply(var13512, var13627)
  #[1,32]
  var13629=tf.reshape(var13628, [1,32])
  #[1,2]
  var13630=tf.matmul(var13629, var13588)
  #[2]
  var13631=tf.reshape(var13630, [2])
  #[2]
  var13632=tf.add(var13631, var13591)
  #[1,2]
  var13633=tf.reshape(var13632, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13634=tf.multiply(var13516, var13627)
  #[1,32]
  var13635=tf.reshape(var13634, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13636=var13569[3:4]
  #[]
  var13637=tf.reshape(var13636, [])
  #[150]
  var13638=tf.gather(params=var13568, indices=var13637, batch_dims=0, axis=0)
  #[150]
  var13639=tf.multiply(var13567, var13638)
  #[1024]
  var13640=tf.gather(params=var13639, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13641=tf.where(var13559, var13640, var13578)
  #[32,32]
  var13642=tf.reshape(var13641, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13643=tf.transpose(var13642, perm=[1,0])
  #[32,32]
  var13644=tf.subtract(var13642, var13643)
  #[32,32]
  var13645=tf.linalg.expm(var13644)
  #[1,32]
  var13646=tf.matmul(var13635, var13645)
  #[32]
  var13647=tf.reshape(var13646, [32])
  #[32]
  var13648=tf.multiply(var13512, var13647)
  #[1,32]
  var13649=tf.reshape(var13648, [1,32])
  #[1,2]
  var13650=tf.matmul(var13649, var13588)
  #[2]
  var13651=tf.reshape(var13650, [2])
  #[2]
  var13652=tf.add(var13651, var13591)
  #[1,2]
  var13653=tf.reshape(var13652, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13654=tf.multiply(var13516, var13647)
  #[1,32]
  var13655=tf.reshape(var13654, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13656=var13569[4:5]
  #[]
  var13657=tf.reshape(var13656, [])
  #[150]
  var13658=tf.gather(params=var13568, indices=var13657, batch_dims=0, axis=0)
  #[150]
  var13659=tf.multiply(var13567, var13658)
  #[1024]
  var13660=tf.gather(params=var13659, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13661=tf.where(var13559, var13660, var13578)
  #[32,32]
  var13662=tf.reshape(var13661, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13663=tf.transpose(var13662, perm=[1,0])
  #[32,32]
  var13664=tf.subtract(var13662, var13663)
  #[32,32]
  var13665=tf.linalg.expm(var13664)
  #[1,32]
  var13666=tf.matmul(var13655, var13665)
  #[32]
  var13667=tf.reshape(var13666, [32])
  #[32]
  var13668=tf.multiply(var13512, var13667)
  #[1,32]
  var13669=tf.reshape(var13668, [1,32])
  #[1,2]
  var13670=tf.matmul(var13669, var13588)
  #[2]
  var13671=tf.reshape(var13670, [2])
  #[2]
  var13672=tf.add(var13671, var13591)
  #[1,2]
  var13673=tf.reshape(var13672, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13674=tf.multiply(var13516, var13667)
  #[1,32]
  var13675=tf.reshape(var13674, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13676=var13569[5:6]
  #[]
  var13677=tf.reshape(var13676, [])
  #[150]
  var13678=tf.gather(params=var13568, indices=var13677, batch_dims=0, axis=0)
  #[150]
  var13679=tf.multiply(var13567, var13678)
  #[1024]
  var13680=tf.gather(params=var13679, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13681=tf.where(var13559, var13680, var13578)
  #[32,32]
  var13682=tf.reshape(var13681, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13683=tf.transpose(var13682, perm=[1,0])
  #[32,32]
  var13684=tf.subtract(var13682, var13683)
  #[32,32]
  var13685=tf.linalg.expm(var13684)
  #[1,32]
  var13686=tf.matmul(var13675, var13685)
  #[32]
  var13687=tf.reshape(var13686, [32])
  #[32]
  var13688=tf.multiply(var13512, var13687)
  #[1,32]
  var13689=tf.reshape(var13688, [1,32])
  #[1,2]
  var13690=tf.matmul(var13689, var13588)
  #[2]
  var13691=tf.reshape(var13690, [2])
  #[2]
  var13692=tf.add(var13691, var13591)
  #[1,2]
  var13693=tf.reshape(var13692, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13694=tf.multiply(var13516, var13687)
  #[1,32]
  var13695=tf.reshape(var13694, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13696=var13569[6:7]
  #[]
  var13697=tf.reshape(var13696, [])
  #[150]
  var13698=tf.gather(params=var13568, indices=var13697, batch_dims=0, axis=0)
  #[150]
  var13699=tf.multiply(var13567, var13698)
  #[1024]
  var13700=tf.gather(params=var13699, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13701=tf.where(var13559, var13700, var13578)
  #[32,32]
  var13702=tf.reshape(var13701, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13703=tf.transpose(var13702, perm=[1,0])
  #[32,32]
  var13704=tf.subtract(var13702, var13703)
  #[32,32]
  var13705=tf.linalg.expm(var13704)
  #[1,32]
  var13706=tf.matmul(var13695, var13705)
  #[32]
  var13707=tf.reshape(var13706, [32])
  #[32]
  var13708=tf.multiply(var13512, var13707)
  #[1,32]
  var13709=tf.reshape(var13708, [1,32])
  #[1,2]
  var13710=tf.matmul(var13709, var13588)
  #[2]
  var13711=tf.reshape(var13710, [2])
  #[2]
  var13712=tf.add(var13711, var13591)
  #[1,2]
  var13713=tf.reshape(var13712, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13714=tf.multiply(var13516, var13707)
  #[1,32]
  var13715=tf.reshape(var13714, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13716=var13569[7:8]
  #[]
  var13717=tf.reshape(var13716, [])
  #[150]
  var13718=tf.gather(params=var13568, indices=var13717, batch_dims=0, axis=0)
  #[150]
  var13719=tf.multiply(var13567, var13718)
  #[1024]
  var13720=tf.gather(params=var13719, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13721=tf.where(var13559, var13720, var13578)
  #[32,32]
  var13722=tf.reshape(var13721, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13723=tf.transpose(var13722, perm=[1,0])
  #[32,32]
  var13724=tf.subtract(var13722, var13723)
  #[32,32]
  var13725=tf.linalg.expm(var13724)
  #[1,32]
  var13726=tf.matmul(var13715, var13725)
  #[32]
  var13727=tf.reshape(var13726, [32])
  #[32]
  var13728=tf.multiply(var13512, var13727)
  #[1,32]
  var13729=tf.reshape(var13728, [1,32])
  #[1,2]
  var13730=tf.matmul(var13729, var13588)
  #[2]
  var13731=tf.reshape(var13730, [2])
  #[2]
  var13732=tf.add(var13731, var13591)
  #[1,2]
  var13733=tf.reshape(var13732, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13734=tf.multiply(var13516, var13727)
  #[1,32]
  var13735=tf.reshape(var13734, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13736=var13569[8:9]
  #[]
  var13737=tf.reshape(var13736, [])
  #[150]
  var13738=tf.gather(params=var13568, indices=var13737, batch_dims=0, axis=0)
  #[150]
  var13739=tf.multiply(var13567, var13738)
  #[1024]
  var13740=tf.gather(params=var13739, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13741=tf.where(var13559, var13740, var13578)
  #[32,32]
  var13742=tf.reshape(var13741, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13743=tf.transpose(var13742, perm=[1,0])
  #[32,32]
  var13744=tf.subtract(var13742, var13743)
  #[32,32]
  var13745=tf.linalg.expm(var13744)
  #[1,32]
  var13746=tf.matmul(var13735, var13745)
  #[32]
  var13747=tf.reshape(var13746, [32])
  #[32]
  var13748=tf.multiply(var13512, var13747)
  #[1,32]
  var13749=tf.reshape(var13748, [1,32])
  #[1,2]
  var13750=tf.matmul(var13749, var13588)
  #[2]
  var13751=tf.reshape(var13750, [2])
  #[2]
  var13752=tf.add(var13751, var13591)
  #[1,2]
  var13753=tf.reshape(var13752, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13754=tf.multiply(var13516, var13747)
  #[1,32]
  var13755=tf.reshape(var13754, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13756=var13569[9:10]
  #[]
  var13757=tf.reshape(var13756, [])
  #[150]
  var13758=tf.gather(params=var13568, indices=var13757, batch_dims=0, axis=0)
  #[150]
  var13759=tf.multiply(var13567, var13758)
  #[1024]
  var13760=tf.gather(params=var13759, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13761=tf.where(var13559, var13760, var13578)
  #[32,32]
  var13762=tf.reshape(var13761, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13763=tf.transpose(var13762, perm=[1,0])
  #[32,32]
  var13764=tf.subtract(var13762, var13763)
  #[32,32]
  var13765=tf.linalg.expm(var13764)
  #[1,32]
  var13766=tf.matmul(var13755, var13765)
  #[32]
  var13767=tf.reshape(var13766, [32])
  #[32]
  var13768=tf.multiply(var13512, var13767)
  #[1,32]
  var13769=tf.reshape(var13768, [1,32])
  #[1,2]
  var13770=tf.matmul(var13769, var13588)
  #[2]
  var13771=tf.reshape(var13770, [2])
  #[2]
  var13772=tf.add(var13771, var13591)
  #[1,2]
  var13773=tf.reshape(var13772, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13774=tf.multiply(var13516, var13767)
  #[1,32]
  var13775=tf.reshape(var13774, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13776=var13569[10:11]
  #[]
  var13777=tf.reshape(var13776, [])
  #[150]
  var13778=tf.gather(params=var13568, indices=var13777, batch_dims=0, axis=0)
  #[150]
  var13779=tf.multiply(var13567, var13778)
  #[1024]
  var13780=tf.gather(params=var13779, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13781=tf.where(var13559, var13780, var13578)
  #[32,32]
  var13782=tf.reshape(var13781, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13783=tf.transpose(var13782, perm=[1,0])
  #[32,32]
  var13784=tf.subtract(var13782, var13783)
  #[32,32]
  var13785=tf.linalg.expm(var13784)
  #[1,32]
  var13786=tf.matmul(var13775, var13785)
  #[32]
  var13787=tf.reshape(var13786, [32])
  #[32]
  var13788=tf.multiply(var13512, var13787)
  #[1,32]
  var13789=tf.reshape(var13788, [1,32])
  #[1,2]
  var13790=tf.matmul(var13789, var13588)
  #[2]
  var13791=tf.reshape(var13790, [2])
  #[2]
  var13792=tf.add(var13791, var13591)
  #[1,2]
  var13793=tf.reshape(var13792, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13794=tf.multiply(var13516, var13787)
  #[1,32]
  var13795=tf.reshape(var13794, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13796=var13569[11:12]
  #[]
  var13797=tf.reshape(var13796, [])
  #[150]
  var13798=tf.gather(params=var13568, indices=var13797, batch_dims=0, axis=0)
  #[150]
  var13799=tf.multiply(var13567, var13798)
  #[1024]
  var13800=tf.gather(params=var13799, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13801=tf.where(var13559, var13800, var13578)
  #[32,32]
  var13802=tf.reshape(var13801, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13803=tf.transpose(var13802, perm=[1,0])
  #[32,32]
  var13804=tf.subtract(var13802, var13803)
  #[32,32]
  var13805=tf.linalg.expm(var13804)
  #[1,32]
  var13806=tf.matmul(var13795, var13805)
  #[32]
  var13807=tf.reshape(var13806, [32])
  #[32]
  var13808=tf.multiply(var13512, var13807)
  #[1,32]
  var13809=tf.reshape(var13808, [1,32])
  #[1,2]
  var13810=tf.matmul(var13809, var13588)
  #[2]
  var13811=tf.reshape(var13810, [2])
  #[2]
  var13812=tf.add(var13811, var13591)
  #[1,2]
  var13813=tf.reshape(var13812, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13814=tf.multiply(var13516, var13807)
  #[1,32]
  var13815=tf.reshape(var13814, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13816=var13569[12:13]
  #[]
  var13817=tf.reshape(var13816, [])
  #[150]
  var13818=tf.gather(params=var13568, indices=var13817, batch_dims=0, axis=0)
  #[150]
  var13819=tf.multiply(var13567, var13818)
  #[1024]
  var13820=tf.gather(params=var13819, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13821=tf.where(var13559, var13820, var13578)
  #[32,32]
  var13822=tf.reshape(var13821, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13823=tf.transpose(var13822, perm=[1,0])
  #[32,32]
  var13824=tf.subtract(var13822, var13823)
  #[32,32]
  var13825=tf.linalg.expm(var13824)
  #[1,32]
  var13826=tf.matmul(var13815, var13825)
  #[32]
  var13827=tf.reshape(var13826, [32])
  #[32]
  var13828=tf.multiply(var13512, var13827)
  #[1,32]
  var13829=tf.reshape(var13828, [1,32])
  #[1,2]
  var13830=tf.matmul(var13829, var13588)
  #[2]
  var13831=tf.reshape(var13830, [2])
  #[2]
  var13832=tf.add(var13831, var13591)
  #[1,2]
  var13833=tf.reshape(var13832, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13834=tf.multiply(var13516, var13827)
  #[1,32]
  var13835=tf.reshape(var13834, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13836=var13569[13:14]
  #[]
  var13837=tf.reshape(var13836, [])
  #[150]
  var13838=tf.gather(params=var13568, indices=var13837, batch_dims=0, axis=0)
  #[150]
  var13839=tf.multiply(var13567, var13838)
  #[1024]
  var13840=tf.gather(params=var13839, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13841=tf.where(var13559, var13840, var13578)
  #[32,32]
  var13842=tf.reshape(var13841, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13843=tf.transpose(var13842, perm=[1,0])
  #[32,32]
  var13844=tf.subtract(var13842, var13843)
  #[32,32]
  var13845=tf.linalg.expm(var13844)
  #[1,32]
  var13846=tf.matmul(var13835, var13845)
  #[32]
  var13847=tf.reshape(var13846, [32])
  #[32]
  var13848=tf.multiply(var13512, var13847)
  #[1,32]
  var13849=tf.reshape(var13848, [1,32])
  #[1,2]
  var13850=tf.matmul(var13849, var13588)
  #[2]
  var13851=tf.reshape(var13850, [2])
  #[2]
  var13852=tf.add(var13851, var13591)
  #[1,2]
  var13853=tf.reshape(var13852, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13854=tf.multiply(var13516, var13847)
  #[1,32]
  var13855=tf.reshape(var13854, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13856=var13569[14:15]
  #[]
  var13857=tf.reshape(var13856, [])
  #[150]
  var13858=tf.gather(params=var13568, indices=var13857, batch_dims=0, axis=0)
  #[150]
  var13859=tf.multiply(var13567, var13858)
  #[1024]
  var13860=tf.gather(params=var13859, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13861=tf.where(var13559, var13860, var13578)
  #[32,32]
  var13862=tf.reshape(var13861, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13863=tf.transpose(var13862, perm=[1,0])
  #[32,32]
  var13864=tf.subtract(var13862, var13863)
  #[32,32]
  var13865=tf.linalg.expm(var13864)
  #[1,32]
  var13866=tf.matmul(var13855, var13865)
  #[32]
  var13867=tf.reshape(var13866, [32])
  #[32]
  var13868=tf.multiply(var13512, var13867)
  #[1,32]
  var13869=tf.reshape(var13868, [1,32])
  #[1,2]
  var13870=tf.matmul(var13869, var13588)
  #[2]
  var13871=tf.reshape(var13870, [2])
  #[2]
  var13872=tf.add(var13871, var13591)
  #[1,2]
  var13873=tf.reshape(var13872, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13874=tf.multiply(var13516, var13867)
  #[1,32]
  var13875=tf.reshape(var13874, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13876=var13569[15:16]
  #[]
  var13877=tf.reshape(var13876, [])
  #[150]
  var13878=tf.gather(params=var13568, indices=var13877, batch_dims=0, axis=0)
  #[150]
  var13879=tf.multiply(var13567, var13878)
  #[1024]
  var13880=tf.gather(params=var13879, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13881=tf.where(var13559, var13880, var13578)
  #[32,32]
  var13882=tf.reshape(var13881, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13883=tf.transpose(var13882, perm=[1,0])
  #[32,32]
  var13884=tf.subtract(var13882, var13883)
  #[32,32]
  var13885=tf.linalg.expm(var13884)
  #[1,32]
  var13886=tf.matmul(var13875, var13885)
  #[32]
  var13887=tf.reshape(var13886, [32])
  #[32]
  var13888=tf.multiply(var13512, var13887)
  #[1,32]
  var13889=tf.reshape(var13888, [1,32])
  #[1,2]
  var13890=tf.matmul(var13889, var13588)
  #[2]
  var13891=tf.reshape(var13890, [2])
  #[2]
  var13892=tf.add(var13891, var13591)
  #[1,2]
  var13893=tf.reshape(var13892, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13894=tf.multiply(var13516, var13887)
  #[1,32]
  var13895=tf.reshape(var13894, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13896=var13569[16:17]
  #[]
  var13897=tf.reshape(var13896, [])
  #[150]
  var13898=tf.gather(params=var13568, indices=var13897, batch_dims=0, axis=0)
  #[150]
  var13899=tf.multiply(var13567, var13898)
  #[1024]
  var13900=tf.gather(params=var13899, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13901=tf.where(var13559, var13900, var13578)
  #[32,32]
  var13902=tf.reshape(var13901, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13903=tf.transpose(var13902, perm=[1,0])
  #[32,32]
  var13904=tf.subtract(var13902, var13903)
  #[32,32]
  var13905=tf.linalg.expm(var13904)
  #[1,32]
  var13906=tf.matmul(var13895, var13905)
  #[32]
  var13907=tf.reshape(var13906, [32])
  #[32]
  var13908=tf.multiply(var13512, var13907)
  #[1,32]
  var13909=tf.reshape(var13908, [1,32])
  #[1,2]
  var13910=tf.matmul(var13909, var13588)
  #[2]
  var13911=tf.reshape(var13910, [2])
  #[2]
  var13912=tf.add(var13911, var13591)
  #[1,2]
  var13913=tf.reshape(var13912, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13914=tf.multiply(var13516, var13907)
  #[1,32]
  var13915=tf.reshape(var13914, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13916=var13569[17:18]
  #[]
  var13917=tf.reshape(var13916, [])
  #[150]
  var13918=tf.gather(params=var13568, indices=var13917, batch_dims=0, axis=0)
  #[150]
  var13919=tf.multiply(var13567, var13918)
  #[1024]
  var13920=tf.gather(params=var13919, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13921=tf.where(var13559, var13920, var13578)
  #[32,32]
  var13922=tf.reshape(var13921, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13923=tf.transpose(var13922, perm=[1,0])
  #[32,32]
  var13924=tf.subtract(var13922, var13923)
  #[32,32]
  var13925=tf.linalg.expm(var13924)
  #[1,32]
  var13926=tf.matmul(var13915, var13925)
  #[32]
  var13927=tf.reshape(var13926, [32])
  #[32]
  var13928=tf.multiply(var13512, var13927)
  #[1,32]
  var13929=tf.reshape(var13928, [1,32])
  #[1,2]
  var13930=tf.matmul(var13929, var13588)
  #[2]
  var13931=tf.reshape(var13930, [2])
  #[2]
  var13932=tf.add(var13931, var13591)
  #[1,2]
  var13933=tf.reshape(var13932, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13934=tf.multiply(var13516, var13927)
  #[1,32]
  var13935=tf.reshape(var13934, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13936=var13569[18:19]
  #[]
  var13937=tf.reshape(var13936, [])
  #[150]
  var13938=tf.gather(params=var13568, indices=var13937, batch_dims=0, axis=0)
  #[150]
  var13939=tf.multiply(var13567, var13938)
  #[1024]
  var13940=tf.gather(params=var13939, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13941=tf.where(var13559, var13940, var13578)
  #[32,32]
  var13942=tf.reshape(var13941, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13943=tf.transpose(var13942, perm=[1,0])
  #[32,32]
  var13944=tf.subtract(var13942, var13943)
  #[32,32]
  var13945=tf.linalg.expm(var13944)
  #[1,32]
  var13946=tf.matmul(var13935, var13945)
  #[32]
  var13947=tf.reshape(var13946, [32])
  #[32]
  var13948=tf.multiply(var13512, var13947)
  #[1,32]
  var13949=tf.reshape(var13948, [1,32])
  #[1,2]
  var13950=tf.matmul(var13949, var13588)
  #[2]
  var13951=tf.reshape(var13950, [2])
  #[2]
  var13952=tf.add(var13951, var13591)
  #[1,2]
  var13953=tf.reshape(var13952, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13954=tf.multiply(var13516, var13947)
  #[1,32]
  var13955=tf.reshape(var13954, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13956=var13569[19:20]
  #[]
  var13957=tf.reshape(var13956, [])
  #[150]
  var13958=tf.gather(params=var13568, indices=var13957, batch_dims=0, axis=0)
  #[150]
  var13959=tf.multiply(var13567, var13958)
  #[1024]
  var13960=tf.gather(params=var13959, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13961=tf.where(var13559, var13960, var13578)
  #[32,32]
  var13962=tf.reshape(var13961, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13963=tf.transpose(var13962, perm=[1,0])
  #[32,32]
  var13964=tf.subtract(var13962, var13963)
  #[32,32]
  var13965=tf.linalg.expm(var13964)
  #[1,32]
  var13966=tf.matmul(var13955, var13965)
  #[32]
  var13967=tf.reshape(var13966, [32])
  #[32]
  var13968=tf.multiply(var13512, var13967)
  #[1,32]
  var13969=tf.reshape(var13968, [1,32])
  #[1,2]
  var13970=tf.matmul(var13969, var13588)
  #[2]
  var13971=tf.reshape(var13970, [2])
  #[2]
  var13972=tf.add(var13971, var13591)
  #[1,2]
  var13973=tf.reshape(var13972, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13974=tf.multiply(var13516, var13967)
  #[1,32]
  var13975=tf.reshape(var13974, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13976=var13569[20:21]
  #[]
  var13977=tf.reshape(var13976, [])
  #[150]
  var13978=tf.gather(params=var13568, indices=var13977, batch_dims=0, axis=0)
  #[150]
  var13979=tf.multiply(var13567, var13978)
  #[1024]
  var13980=tf.gather(params=var13979, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var13981=tf.where(var13559, var13980, var13578)
  #[32,32]
  var13982=tf.reshape(var13981, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var13983=tf.transpose(var13982, perm=[1,0])
  #[32,32]
  var13984=tf.subtract(var13982, var13983)
  #[32,32]
  var13985=tf.linalg.expm(var13984)
  #[1,32]
  var13986=tf.matmul(var13975, var13985)
  #[32]
  var13987=tf.reshape(var13986, [32])
  #[32]
  var13988=tf.multiply(var13512, var13987)
  #[1,32]
  var13989=tf.reshape(var13988, [1,32])
  #[1,2]
  var13990=tf.matmul(var13989, var13588)
  #[2]
  var13991=tf.reshape(var13990, [2])
  #[2]
  var13992=tf.add(var13991, var13591)
  #[1,2]
  var13993=tf.reshape(var13992, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var13994=tf.multiply(var13516, var13987)
  #[1,32]
  var13995=tf.reshape(var13994, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var13996=var13569[21:22]
  #[]
  var13997=tf.reshape(var13996, [])
  #[150]
  var13998=tf.gather(params=var13568, indices=var13997, batch_dims=0, axis=0)
  #[150]
  var13999=tf.multiply(var13567, var13998)
  #[1024]
  var14000=tf.gather(params=var13999, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14001=tf.where(var13559, var14000, var13578)
  #[32,32]
  var14002=tf.reshape(var14001, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14003=tf.transpose(var14002, perm=[1,0])
  #[32,32]
  var14004=tf.subtract(var14002, var14003)
  #[32,32]
  var14005=tf.linalg.expm(var14004)
  #[1,32]
  var14006=tf.matmul(var13995, var14005)
  #[32]
  var14007=tf.reshape(var14006, [32])
  #[32]
  var14008=tf.multiply(var13512, var14007)
  #[1,32]
  var14009=tf.reshape(var14008, [1,32])
  #[1,2]
  var14010=tf.matmul(var14009, var13588)
  #[2]
  var14011=tf.reshape(var14010, [2])
  #[2]
  var14012=tf.add(var14011, var13591)
  #[1,2]
  var14013=tf.reshape(var14012, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14014=tf.multiply(var13516, var14007)
  #[1,32]
  var14015=tf.reshape(var14014, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14016=var13569[22:23]
  #[]
  var14017=tf.reshape(var14016, [])
  #[150]
  var14018=tf.gather(params=var13568, indices=var14017, batch_dims=0, axis=0)
  #[150]
  var14019=tf.multiply(var13567, var14018)
  #[1024]
  var14020=tf.gather(params=var14019, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14021=tf.where(var13559, var14020, var13578)
  #[32,32]
  var14022=tf.reshape(var14021, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14023=tf.transpose(var14022, perm=[1,0])
  #[32,32]
  var14024=tf.subtract(var14022, var14023)
  #[32,32]
  var14025=tf.linalg.expm(var14024)
  #[1,32]
  var14026=tf.matmul(var14015, var14025)
  #[32]
  var14027=tf.reshape(var14026, [32])
  #[32]
  var14028=tf.multiply(var13512, var14027)
  #[1,32]
  var14029=tf.reshape(var14028, [1,32])
  #[1,2]
  var14030=tf.matmul(var14029, var13588)
  #[2]
  var14031=tf.reshape(var14030, [2])
  #[2]
  var14032=tf.add(var14031, var13591)
  #[1,2]
  var14033=tf.reshape(var14032, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14034=tf.multiply(var13516, var14027)
  #[1,32]
  var14035=tf.reshape(var14034, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14036=var13569[23:24]
  #[]
  var14037=tf.reshape(var14036, [])
  #[150]
  var14038=tf.gather(params=var13568, indices=var14037, batch_dims=0, axis=0)
  #[150]
  var14039=tf.multiply(var13567, var14038)
  #[1024]
  var14040=tf.gather(params=var14039, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14041=tf.where(var13559, var14040, var13578)
  #[32,32]
  var14042=tf.reshape(var14041, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14043=tf.transpose(var14042, perm=[1,0])
  #[32,32]
  var14044=tf.subtract(var14042, var14043)
  #[32,32]
  var14045=tf.linalg.expm(var14044)
  #[1,32]
  var14046=tf.matmul(var14035, var14045)
  #[32]
  var14047=tf.reshape(var14046, [32])
  #[32]
  var14048=tf.multiply(var13512, var14047)
  #[1,32]
  var14049=tf.reshape(var14048, [1,32])
  #[1,2]
  var14050=tf.matmul(var14049, var13588)
  #[2]
  var14051=tf.reshape(var14050, [2])
  #[2]
  var14052=tf.add(var14051, var13591)
  #[1,2]
  var14053=tf.reshape(var14052, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14054=tf.multiply(var13516, var14047)
  #[1,32]
  var14055=tf.reshape(var14054, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14056=var13569[24:25]
  #[]
  var14057=tf.reshape(var14056, [])
  #[150]
  var14058=tf.gather(params=var13568, indices=var14057, batch_dims=0, axis=0)
  #[150]
  var14059=tf.multiply(var13567, var14058)
  #[1024]
  var14060=tf.gather(params=var14059, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14061=tf.where(var13559, var14060, var13578)
  #[32,32]
  var14062=tf.reshape(var14061, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14063=tf.transpose(var14062, perm=[1,0])
  #[32,32]
  var14064=tf.subtract(var14062, var14063)
  #[32,32]
  var14065=tf.linalg.expm(var14064)
  #[1,32]
  var14066=tf.matmul(var14055, var14065)
  #[32]
  var14067=tf.reshape(var14066, [32])
  #[32]
  var14068=tf.multiply(var13512, var14067)
  #[1,32]
  var14069=tf.reshape(var14068, [1,32])
  #[1,2]
  var14070=tf.matmul(var14069, var13588)
  #[2]
  var14071=tf.reshape(var14070, [2])
  #[2]
  var14072=tf.add(var14071, var13591)
  #[1,2]
  var14073=tf.reshape(var14072, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14074=tf.multiply(var13516, var14067)
  #[1,32]
  var14075=tf.reshape(var14074, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14076=var13569[25:26]
  #[]
  var14077=tf.reshape(var14076, [])
  #[150]
  var14078=tf.gather(params=var13568, indices=var14077, batch_dims=0, axis=0)
  #[150]
  var14079=tf.multiply(var13567, var14078)
  #[1024]
  var14080=tf.gather(params=var14079, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14081=tf.where(var13559, var14080, var13578)
  #[32,32]
  var14082=tf.reshape(var14081, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14083=tf.transpose(var14082, perm=[1,0])
  #[32,32]
  var14084=tf.subtract(var14082, var14083)
  #[32,32]
  var14085=tf.linalg.expm(var14084)
  #[1,32]
  var14086=tf.matmul(var14075, var14085)
  #[32]
  var14087=tf.reshape(var14086, [32])
  #[32]
  var14088=tf.multiply(var13512, var14087)
  #[1,32]
  var14089=tf.reshape(var14088, [1,32])
  #[1,2]
  var14090=tf.matmul(var14089, var13588)
  #[2]
  var14091=tf.reshape(var14090, [2])
  #[2]
  var14092=tf.add(var14091, var13591)
  #[1,2]
  var14093=tf.reshape(var14092, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14094=tf.multiply(var13516, var14087)
  #[1,32]
  var14095=tf.reshape(var14094, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14096=var13569[26:27]
  #[]
  var14097=tf.reshape(var14096, [])
  #[150]
  var14098=tf.gather(params=var13568, indices=var14097, batch_dims=0, axis=0)
  #[150]
  var14099=tf.multiply(var13567, var14098)
  #[1024]
  var14100=tf.gather(params=var14099, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14101=tf.where(var13559, var14100, var13578)
  #[32,32]
  var14102=tf.reshape(var14101, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14103=tf.transpose(var14102, perm=[1,0])
  #[32,32]
  var14104=tf.subtract(var14102, var14103)
  #[32,32]
  var14105=tf.linalg.expm(var14104)
  #[1,32]
  var14106=tf.matmul(var14095, var14105)
  #[32]
  var14107=tf.reshape(var14106, [32])
  #[32]
  var14108=tf.multiply(var13512, var14107)
  #[1,32]
  var14109=tf.reshape(var14108, [1,32])
  #[1,2]
  var14110=tf.matmul(var14109, var13588)
  #[2]
  var14111=tf.reshape(var14110, [2])
  #[2]
  var14112=tf.add(var14111, var13591)
  #[1,2]
  var14113=tf.reshape(var14112, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14114=tf.multiply(var13516, var14107)
  #[1,32]
  var14115=tf.reshape(var14114, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14116=var13569[27:28]
  #[]
  var14117=tf.reshape(var14116, [])
  #[150]
  var14118=tf.gather(params=var13568, indices=var14117, batch_dims=0, axis=0)
  #[150]
  var14119=tf.multiply(var13567, var14118)
  #[1024]
  var14120=tf.gather(params=var14119, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14121=tf.where(var13559, var14120, var13578)
  #[32,32]
  var14122=tf.reshape(var14121, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14123=tf.transpose(var14122, perm=[1,0])
  #[32,32]
  var14124=tf.subtract(var14122, var14123)
  #[32,32]
  var14125=tf.linalg.expm(var14124)
  #[1,32]
  var14126=tf.matmul(var14115, var14125)
  #[32]
  var14127=tf.reshape(var14126, [32])
  #[32]
  var14128=tf.multiply(var13512, var14127)
  #[1,32]
  var14129=tf.reshape(var14128, [1,32])
  #[1,2]
  var14130=tf.matmul(var14129, var13588)
  #[2]
  var14131=tf.reshape(var14130, [2])
  #[2]
  var14132=tf.add(var14131, var13591)
  #[1,2]
  var14133=tf.reshape(var14132, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14134=tf.multiply(var13516, var14127)
  #[1,32]
  var14135=tf.reshape(var14134, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14136=var13569[28:29]
  #[]
  var14137=tf.reshape(var14136, [])
  #[150]
  var14138=tf.gather(params=var13568, indices=var14137, batch_dims=0, axis=0)
  #[150]
  var14139=tf.multiply(var13567, var14138)
  #[1024]
  var14140=tf.gather(params=var14139, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14141=tf.where(var13559, var14140, var13578)
  #[32,32]
  var14142=tf.reshape(var14141, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14143=tf.transpose(var14142, perm=[1,0])
  #[32,32]
  var14144=tf.subtract(var14142, var14143)
  #[32,32]
  var14145=tf.linalg.expm(var14144)
  #[1,32]
  var14146=tf.matmul(var14135, var14145)
  #[32]
  var14147=tf.reshape(var14146, [32])
  #[32]
  var14148=tf.multiply(var13512, var14147)
  #[1,32]
  var14149=tf.reshape(var14148, [1,32])
  #[1,2]
  var14150=tf.matmul(var14149, var13588)
  #[2]
  var14151=tf.reshape(var14150, [2])
  #[2]
  var14152=tf.add(var14151, var13591)
  #[1,2]
  var14153=tf.reshape(var14152, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14154=tf.multiply(var13516, var14147)
  #[1,32]
  var14155=tf.reshape(var14154, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14156=var13569[29:30]
  #[]
  var14157=tf.reshape(var14156, [])
  #[150]
  var14158=tf.gather(params=var13568, indices=var14157, batch_dims=0, axis=0)
  #[150]
  var14159=tf.multiply(var13567, var14158)
  #[1024]
  var14160=tf.gather(params=var14159, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14161=tf.where(var13559, var14160, var13578)
  #[32,32]
  var14162=tf.reshape(var14161, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14163=tf.transpose(var14162, perm=[1,0])
  #[32,32]
  var14164=tf.subtract(var14162, var14163)
  #[32,32]
  var14165=tf.linalg.expm(var14164)
  #[1,32]
  var14166=tf.matmul(var14155, var14165)
  #[32]
  var14167=tf.reshape(var14166, [32])
  #[32]
  var14168=tf.multiply(var13512, var14167)
  #[1,32]
  var14169=tf.reshape(var14168, [1,32])
  #[1,2]
  var14170=tf.matmul(var14169, var13588)
  #[2]
  var14171=tf.reshape(var14170, [2])
  #[2]
  var14172=tf.add(var14171, var13591)
  #[1,2]
  var14173=tf.reshape(var14172, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14174=tf.multiply(var13516, var14167)
  #[1,32]
  var14175=tf.reshape(var14174, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14176=var13569[30:31]
  #[]
  var14177=tf.reshape(var14176, [])
  #[150]
  var14178=tf.gather(params=var13568, indices=var14177, batch_dims=0, axis=0)
  #[150]
  var14179=tf.multiply(var13567, var14178)
  #[1024]
  var14180=tf.gather(params=var14179, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14181=tf.where(var13559, var14180, var13578)
  #[32,32]
  var14182=tf.reshape(var14181, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14183=tf.transpose(var14182, perm=[1,0])
  #[32,32]
  var14184=tf.subtract(var14182, var14183)
  #[32,32]
  var14185=tf.linalg.expm(var14184)
  #[1,32]
  var14186=tf.matmul(var14175, var14185)
  #[32]
  var14187=tf.reshape(var14186, [32])
  #[32]
  var14188=tf.multiply(var13512, var14187)
  #[1,32]
  var14189=tf.reshape(var14188, [1,32])
  #[1,2]
  var14190=tf.matmul(var14189, var13588)
  #[2]
  var14191=tf.reshape(var14190, [2])
  #[2]
  var14192=tf.add(var14191, var13591)
  #[1,2]
  var14193=tf.reshape(var14192, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14194=tf.multiply(var13516, var14187)
  #[1,32]
  var14195=tf.reshape(var14194, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14196=var13569[31:32]
  #[]
  var14197=tf.reshape(var14196, [])
  #[150]
  var14198=tf.gather(params=var13568, indices=var14197, batch_dims=0, axis=0)
  #[150]
  var14199=tf.multiply(var13567, var14198)
  #[1024]
  var14200=tf.gather(params=var14199, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14201=tf.where(var13559, var14200, var13578)
  #[32,32]
  var14202=tf.reshape(var14201, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14203=tf.transpose(var14202, perm=[1,0])
  #[32,32]
  var14204=tf.subtract(var14202, var14203)
  #[32,32]
  var14205=tf.linalg.expm(var14204)
  #[1,32]
  var14206=tf.matmul(var14195, var14205)
  #[32]
  var14207=tf.reshape(var14206, [32])
  #[32]
  var14208=tf.multiply(var13512, var14207)
  #[1,32]
  var14209=tf.reshape(var14208, [1,32])
  #[1,2]
  var14210=tf.matmul(var14209, var13588)
  #[2]
  var14211=tf.reshape(var14210, [2])
  #[2]
  var14212=tf.add(var14211, var13591)
  #[1,2]
  var14213=tf.reshape(var14212, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14214=tf.multiply(var13516, var14207)
  #[1,32]
  var14215=tf.reshape(var14214, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14216=var13569[32:33]
  #[]
  var14217=tf.reshape(var14216, [])
  #[150]
  var14218=tf.gather(params=var13568, indices=var14217, batch_dims=0, axis=0)
  #[150]
  var14219=tf.multiply(var13567, var14218)
  #[1024]
  var14220=tf.gather(params=var14219, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14221=tf.where(var13559, var14220, var13578)
  #[32,32]
  var14222=tf.reshape(var14221, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14223=tf.transpose(var14222, perm=[1,0])
  #[32,32]
  var14224=tf.subtract(var14222, var14223)
  #[32,32]
  var14225=tf.linalg.expm(var14224)
  #[1,32]
  var14226=tf.matmul(var14215, var14225)
  #[32]
  var14227=tf.reshape(var14226, [32])
  #[32]
  var14228=tf.multiply(var13512, var14227)
  #[1,32]
  var14229=tf.reshape(var14228, [1,32])
  #[1,2]
  var14230=tf.matmul(var14229, var13588)
  #[2]
  var14231=tf.reshape(var14230, [2])
  #[2]
  var14232=tf.add(var14231, var13591)
  #[1,2]
  var14233=tf.reshape(var14232, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14234=tf.multiply(var13516, var14227)
  #[1,32]
  var14235=tf.reshape(var14234, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14236=var13569[33:34]
  #[]
  var14237=tf.reshape(var14236, [])
  #[150]
  var14238=tf.gather(params=var13568, indices=var14237, batch_dims=0, axis=0)
  #[150]
  var14239=tf.multiply(var13567, var14238)
  #[1024]
  var14240=tf.gather(params=var14239, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14241=tf.where(var13559, var14240, var13578)
  #[32,32]
  var14242=tf.reshape(var14241, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14243=tf.transpose(var14242, perm=[1,0])
  #[32,32]
  var14244=tf.subtract(var14242, var14243)
  #[32,32]
  var14245=tf.linalg.expm(var14244)
  #[1,32]
  var14246=tf.matmul(var14235, var14245)
  #[32]
  var14247=tf.reshape(var14246, [32])
  #[32]
  var14248=tf.multiply(var13512, var14247)
  #[1,32]
  var14249=tf.reshape(var14248, [1,32])
  #[1,2]
  var14250=tf.matmul(var14249, var13588)
  #[2]
  var14251=tf.reshape(var14250, [2])
  #[2]
  var14252=tf.add(var14251, var13591)
  #[1,2]
  var14253=tf.reshape(var14252, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14254=tf.multiply(var13516, var14247)
  #[1,32]
  var14255=tf.reshape(var14254, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14256=var13569[34:35]
  #[]
  var14257=tf.reshape(var14256, [])
  #[150]
  var14258=tf.gather(params=var13568, indices=var14257, batch_dims=0, axis=0)
  #[150]
  var14259=tf.multiply(var13567, var14258)
  #[1024]
  var14260=tf.gather(params=var14259, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14261=tf.where(var13559, var14260, var13578)
  #[32,32]
  var14262=tf.reshape(var14261, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14263=tf.transpose(var14262, perm=[1,0])
  #[32,32]
  var14264=tf.subtract(var14262, var14263)
  #[32,32]
  var14265=tf.linalg.expm(var14264)
  #[1,32]
  var14266=tf.matmul(var14255, var14265)
  #[32]
  var14267=tf.reshape(var14266, [32])
  #[32]
  var14268=tf.multiply(var13512, var14267)
  #[1,32]
  var14269=tf.reshape(var14268, [1,32])
  #[1,2]
  var14270=tf.matmul(var14269, var13588)
  #[2]
  var14271=tf.reshape(var14270, [2])
  #[2]
  var14272=tf.add(var14271, var13591)
  #[1,2]
  var14273=tf.reshape(var14272, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14274=tf.multiply(var13516, var14267)
  #[1,32]
  var14275=tf.reshape(var14274, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14276=var13569[35:36]
  #[]
  var14277=tf.reshape(var14276, [])
  #[150]
  var14278=tf.gather(params=var13568, indices=var14277, batch_dims=0, axis=0)
  #[150]
  var14279=tf.multiply(var13567, var14278)
  #[1024]
  var14280=tf.gather(params=var14279, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14281=tf.where(var13559, var14280, var13578)
  #[32,32]
  var14282=tf.reshape(var14281, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14283=tf.transpose(var14282, perm=[1,0])
  #[32,32]
  var14284=tf.subtract(var14282, var14283)
  #[32,32]
  var14285=tf.linalg.expm(var14284)
  #[1,32]
  var14286=tf.matmul(var14275, var14285)
  #[32]
  var14287=tf.reshape(var14286, [32])
  #[32]
  var14288=tf.multiply(var13512, var14287)
  #[1,32]
  var14289=tf.reshape(var14288, [1,32])
  #[1,2]
  var14290=tf.matmul(var14289, var13588)
  #[2]
  var14291=tf.reshape(var14290, [2])
  #[2]
  var14292=tf.add(var14291, var13591)
  #[1,2]
  var14293=tf.reshape(var14292, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14294=tf.multiply(var13516, var14287)
  #[1,32]
  var14295=tf.reshape(var14294, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14296=var13569[36:37]
  #[]
  var14297=tf.reshape(var14296, [])
  #[150]
  var14298=tf.gather(params=var13568, indices=var14297, batch_dims=0, axis=0)
  #[150]
  var14299=tf.multiply(var13567, var14298)
  #[1024]
  var14300=tf.gather(params=var14299, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14301=tf.where(var13559, var14300, var13578)
  #[32,32]
  var14302=tf.reshape(var14301, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14303=tf.transpose(var14302, perm=[1,0])
  #[32,32]
  var14304=tf.subtract(var14302, var14303)
  #[32,32]
  var14305=tf.linalg.expm(var14304)
  #[1,32]
  var14306=tf.matmul(var14295, var14305)
  #[32]
  var14307=tf.reshape(var14306, [32])
  #[32]
  var14308=tf.multiply(var13512, var14307)
  #[1,32]
  var14309=tf.reshape(var14308, [1,32])
  #[1,2]
  var14310=tf.matmul(var14309, var13588)
  #[2]
  var14311=tf.reshape(var14310, [2])
  #[2]
  var14312=tf.add(var14311, var13591)
  #[1,2]
  var14313=tf.reshape(var14312, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14314=tf.multiply(var13516, var14307)
  #[1,32]
  var14315=tf.reshape(var14314, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14316=var13569[37:38]
  #[]
  var14317=tf.reshape(var14316, [])
  #[150]
  var14318=tf.gather(params=var13568, indices=var14317, batch_dims=0, axis=0)
  #[150]
  var14319=tf.multiply(var13567, var14318)
  #[1024]
  var14320=tf.gather(params=var14319, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14321=tf.where(var13559, var14320, var13578)
  #[32,32]
  var14322=tf.reshape(var14321, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14323=tf.transpose(var14322, perm=[1,0])
  #[32,32]
  var14324=tf.subtract(var14322, var14323)
  #[32,32]
  var14325=tf.linalg.expm(var14324)
  #[1,32]
  var14326=tf.matmul(var14315, var14325)
  #[32]
  var14327=tf.reshape(var14326, [32])
  #[32]
  var14328=tf.multiply(var13512, var14327)
  #[1,32]
  var14329=tf.reshape(var14328, [1,32])
  #[1,2]
  var14330=tf.matmul(var14329, var13588)
  #[2]
  var14331=tf.reshape(var14330, [2])
  #[2]
  var14332=tf.add(var14331, var13591)
  #[1,2]
  var14333=tf.reshape(var14332, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14334=tf.multiply(var13516, var14327)
  #[1,32]
  var14335=tf.reshape(var14334, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14336=var13569[38:39]
  #[]
  var14337=tf.reshape(var14336, [])
  #[150]
  var14338=tf.gather(params=var13568, indices=var14337, batch_dims=0, axis=0)
  #[150]
  var14339=tf.multiply(var13567, var14338)
  #[1024]
  var14340=tf.gather(params=var14339, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14341=tf.where(var13559, var14340, var13578)
  #[32,32]
  var14342=tf.reshape(var14341, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14343=tf.transpose(var14342, perm=[1,0])
  #[32,32]
  var14344=tf.subtract(var14342, var14343)
  #[32,32]
  var14345=tf.linalg.expm(var14344)
  #[1,32]
  var14346=tf.matmul(var14335, var14345)
  #[32]
  var14347=tf.reshape(var14346, [32])
  #[32]
  var14348=tf.multiply(var13512, var14347)
  #[1,32]
  var14349=tf.reshape(var14348, [1,32])
  #[1,2]
  var14350=tf.matmul(var14349, var13588)
  #[2]
  var14351=tf.reshape(var14350, [2])
  #[2]
  var14352=tf.add(var14351, var13591)
  #[1,2]
  var14353=tf.reshape(var14352, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14354=tf.multiply(var13516, var14347)
  #[1,32]
  var14355=tf.reshape(var14354, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14356=var13569[39:40]
  #[]
  var14357=tf.reshape(var14356, [])
  #[150]
  var14358=tf.gather(params=var13568, indices=var14357, batch_dims=0, axis=0)
  #[150]
  var14359=tf.multiply(var13567, var14358)
  #[1024]
  var14360=tf.gather(params=var14359, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14361=tf.where(var13559, var14360, var13578)
  #[32,32]
  var14362=tf.reshape(var14361, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14363=tf.transpose(var14362, perm=[1,0])
  #[32,32]
  var14364=tf.subtract(var14362, var14363)
  #[32,32]
  var14365=tf.linalg.expm(var14364)
  #[1,32]
  var14366=tf.matmul(var14355, var14365)
  #[32]
  var14367=tf.reshape(var14366, [32])
  #[32]
  var14368=tf.multiply(var13512, var14367)
  #[1,32]
  var14369=tf.reshape(var14368, [1,32])
  #[1,2]
  var14370=tf.matmul(var14369, var13588)
  #[2]
  var14371=tf.reshape(var14370, [2])
  #[2]
  var14372=tf.add(var14371, var13591)
  #[1,2]
  var14373=tf.reshape(var14372, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14374=tf.multiply(var13516, var14367)
  #[1,32]
  var14375=tf.reshape(var14374, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14376=var13569[40:41]
  #[]
  var14377=tf.reshape(var14376, [])
  #[150]
  var14378=tf.gather(params=var13568, indices=var14377, batch_dims=0, axis=0)
  #[150]
  var14379=tf.multiply(var13567, var14378)
  #[1024]
  var14380=tf.gather(params=var14379, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14381=tf.where(var13559, var14380, var13578)
  #[32,32]
  var14382=tf.reshape(var14381, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14383=tf.transpose(var14382, perm=[1,0])
  #[32,32]
  var14384=tf.subtract(var14382, var14383)
  #[32,32]
  var14385=tf.linalg.expm(var14384)
  #[1,32]
  var14386=tf.matmul(var14375, var14385)
  #[32]
  var14387=tf.reshape(var14386, [32])
  #[32]
  var14388=tf.multiply(var13512, var14387)
  #[1,32]
  var14389=tf.reshape(var14388, [1,32])
  #[1,2]
  var14390=tf.matmul(var14389, var13588)
  #[2]
  var14391=tf.reshape(var14390, [2])
  #[2]
  var14392=tf.add(var14391, var13591)
  #[1,2]
  var14393=tf.reshape(var14392, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14394=tf.multiply(var13516, var14387)
  #[1,32]
  var14395=tf.reshape(var14394, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14396=var13569[41:42]
  #[]
  var14397=tf.reshape(var14396, [])
  #[150]
  var14398=tf.gather(params=var13568, indices=var14397, batch_dims=0, axis=0)
  #[150]
  var14399=tf.multiply(var13567, var14398)
  #[1024]
  var14400=tf.gather(params=var14399, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14401=tf.where(var13559, var14400, var13578)
  #[32,32]
  var14402=tf.reshape(var14401, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14403=tf.transpose(var14402, perm=[1,0])
  #[32,32]
  var14404=tf.subtract(var14402, var14403)
  #[32,32]
  var14405=tf.linalg.expm(var14404)
  #[1,32]
  var14406=tf.matmul(var14395, var14405)
  #[32]
  var14407=tf.reshape(var14406, [32])
  #[32]
  var14408=tf.multiply(var13512, var14407)
  #[1,32]
  var14409=tf.reshape(var14408, [1,32])
  #[1,2]
  var14410=tf.matmul(var14409, var13588)
  #[2]
  var14411=tf.reshape(var14410, [2])
  #[2]
  var14412=tf.add(var14411, var13591)
  #[1,2]
  var14413=tf.reshape(var14412, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14414=tf.multiply(var13516, var14407)
  #[1,32]
  var14415=tf.reshape(var14414, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14416=var13569[42:43]
  #[]
  var14417=tf.reshape(var14416, [])
  #[150]
  var14418=tf.gather(params=var13568, indices=var14417, batch_dims=0, axis=0)
  #[150]
  var14419=tf.multiply(var13567, var14418)
  #[1024]
  var14420=tf.gather(params=var14419, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14421=tf.where(var13559, var14420, var13578)
  #[32,32]
  var14422=tf.reshape(var14421, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14423=tf.transpose(var14422, perm=[1,0])
  #[32,32]
  var14424=tf.subtract(var14422, var14423)
  #[32,32]
  var14425=tf.linalg.expm(var14424)
  #[1,32]
  var14426=tf.matmul(var14415, var14425)
  #[32]
  var14427=tf.reshape(var14426, [32])
  #[32]
  var14428=tf.multiply(var13512, var14427)
  #[1,32]
  var14429=tf.reshape(var14428, [1,32])
  #[1,2]
  var14430=tf.matmul(var14429, var13588)
  #[2]
  var14431=tf.reshape(var14430, [2])
  #[2]
  var14432=tf.add(var14431, var13591)
  #[1,2]
  var14433=tf.reshape(var14432, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14434=tf.multiply(var13516, var14427)
  #[1,32]
  var14435=tf.reshape(var14434, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14436=var13569[43:44]
  #[]
  var14437=tf.reshape(var14436, [])
  #[150]
  var14438=tf.gather(params=var13568, indices=var14437, batch_dims=0, axis=0)
  #[150]
  var14439=tf.multiply(var13567, var14438)
  #[1024]
  var14440=tf.gather(params=var14439, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14441=tf.where(var13559, var14440, var13578)
  #[32,32]
  var14442=tf.reshape(var14441, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14443=tf.transpose(var14442, perm=[1,0])
  #[32,32]
  var14444=tf.subtract(var14442, var14443)
  #[32,32]
  var14445=tf.linalg.expm(var14444)
  #[1,32]
  var14446=tf.matmul(var14435, var14445)
  #[32]
  var14447=tf.reshape(var14446, [32])
  #[32]
  var14448=tf.multiply(var13512, var14447)
  #[1,32]
  var14449=tf.reshape(var14448, [1,32])
  #[1,2]
  var14450=tf.matmul(var14449, var13588)
  #[2]
  var14451=tf.reshape(var14450, [2])
  #[2]
  var14452=tf.add(var14451, var13591)
  #[1,2]
  var14453=tf.reshape(var14452, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14454=tf.multiply(var13516, var14447)
  #[1,32]
  var14455=tf.reshape(var14454, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14456=var13569[44:45]
  #[]
  var14457=tf.reshape(var14456, [])
  #[150]
  var14458=tf.gather(params=var13568, indices=var14457, batch_dims=0, axis=0)
  #[150]
  var14459=tf.multiply(var13567, var14458)
  #[1024]
  var14460=tf.gather(params=var14459, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14461=tf.where(var13559, var14460, var13578)
  #[32,32]
  var14462=tf.reshape(var14461, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14463=tf.transpose(var14462, perm=[1,0])
  #[32,32]
  var14464=tf.subtract(var14462, var14463)
  #[32,32]
  var14465=tf.linalg.expm(var14464)
  #[1,32]
  var14466=tf.matmul(var14455, var14465)
  #[32]
  var14467=tf.reshape(var14466, [32])
  #[32]
  var14468=tf.multiply(var13512, var14467)
  #[1,32]
  var14469=tf.reshape(var14468, [1,32])
  #[1,2]
  var14470=tf.matmul(var14469, var13588)
  #[2]
  var14471=tf.reshape(var14470, [2])
  #[2]
  var14472=tf.add(var14471, var13591)
  #[1,2]
  var14473=tf.reshape(var14472, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14474=tf.multiply(var13516, var14467)
  #[1,32]
  var14475=tf.reshape(var14474, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14476=var13569[45:46]
  #[]
  var14477=tf.reshape(var14476, [])
  #[150]
  var14478=tf.gather(params=var13568, indices=var14477, batch_dims=0, axis=0)
  #[150]
  var14479=tf.multiply(var13567, var14478)
  #[1024]
  var14480=tf.gather(params=var14479, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14481=tf.where(var13559, var14480, var13578)
  #[32,32]
  var14482=tf.reshape(var14481, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14483=tf.transpose(var14482, perm=[1,0])
  #[32,32]
  var14484=tf.subtract(var14482, var14483)
  #[32,32]
  var14485=tf.linalg.expm(var14484)
  #[1,32]
  var14486=tf.matmul(var14475, var14485)
  #[32]
  var14487=tf.reshape(var14486, [32])
  #[32]
  var14488=tf.multiply(var13512, var14487)
  #[1,32]
  var14489=tf.reshape(var14488, [1,32])
  #[1,2]
  var14490=tf.matmul(var14489, var13588)
  #[2]
  var14491=tf.reshape(var14490, [2])
  #[2]
  var14492=tf.add(var14491, var13591)
  #[1,2]
  var14493=tf.reshape(var14492, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14494=tf.multiply(var13516, var14487)
  #[1,32]
  var14495=tf.reshape(var14494, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14496=var13569[46:47]
  #[]
  var14497=tf.reshape(var14496, [])
  #[150]
  var14498=tf.gather(params=var13568, indices=var14497, batch_dims=0, axis=0)
  #[150]
  var14499=tf.multiply(var13567, var14498)
  #[1024]
  var14500=tf.gather(params=var14499, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14501=tf.where(var13559, var14500, var13578)
  #[32,32]
  var14502=tf.reshape(var14501, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14503=tf.transpose(var14502, perm=[1,0])
  #[32,32]
  var14504=tf.subtract(var14502, var14503)
  #[32,32]
  var14505=tf.linalg.expm(var14504)
  #[1,32]
  var14506=tf.matmul(var14495, var14505)
  #[32]
  var14507=tf.reshape(var14506, [32])
  #[32]
  var14508=tf.multiply(var13512, var14507)
  #[1,32]
  var14509=tf.reshape(var14508, [1,32])
  #[1,2]
  var14510=tf.matmul(var14509, var13588)
  #[2]
  var14511=tf.reshape(var14510, [2])
  #[2]
  var14512=tf.add(var14511, var13591)
  #[1,2]
  var14513=tf.reshape(var14512, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14514=tf.multiply(var13516, var14507)
  #[1,32]
  var14515=tf.reshape(var14514, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14516=var13569[47:48]
  #[]
  var14517=tf.reshape(var14516, [])
  #[150]
  var14518=tf.gather(params=var13568, indices=var14517, batch_dims=0, axis=0)
  #[150]
  var14519=tf.multiply(var13567, var14518)
  #[1024]
  var14520=tf.gather(params=var14519, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14521=tf.where(var13559, var14520, var13578)
  #[32,32]
  var14522=tf.reshape(var14521, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14523=tf.transpose(var14522, perm=[1,0])
  #[32,32]
  var14524=tf.subtract(var14522, var14523)
  #[32,32]
  var14525=tf.linalg.expm(var14524)
  #[1,32]
  var14526=tf.matmul(var14515, var14525)
  #[32]
  var14527=tf.reshape(var14526, [32])
  #[32]
  var14528=tf.multiply(var13512, var14527)
  #[1,32]
  var14529=tf.reshape(var14528, [1,32])
  #[1,2]
  var14530=tf.matmul(var14529, var13588)
  #[2]
  var14531=tf.reshape(var14530, [2])
  #[2]
  var14532=tf.add(var14531, var13591)
  #[1,2]
  var14533=tf.reshape(var14532, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14534=tf.multiply(var13516, var14527)
  #[1,32]
  var14535=tf.reshape(var14534, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14536=var13569[48:49]
  #[]
  var14537=tf.reshape(var14536, [])
  #[150]
  var14538=tf.gather(params=var13568, indices=var14537, batch_dims=0, axis=0)
  #[150]
  var14539=tf.multiply(var13567, var14538)
  #[1024]
  var14540=tf.gather(params=var14539, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14541=tf.where(var13559, var14540, var13578)
  #[32,32]
  var14542=tf.reshape(var14541, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14543=tf.transpose(var14542, perm=[1,0])
  #[32,32]
  var14544=tf.subtract(var14542, var14543)
  #[32,32]
  var14545=tf.linalg.expm(var14544)
  #[1,32]
  var14546=tf.matmul(var14535, var14545)
  #[32]
  var14547=tf.reshape(var14546, [32])
  #[32]
  var14548=tf.multiply(var13512, var14547)
  #[1,32]
  var14549=tf.reshape(var14548, [1,32])
  #[1,2]
  var14550=tf.matmul(var14549, var13588)
  #[2]
  var14551=tf.reshape(var14550, [2])
  #[2]
  var14552=tf.add(var14551, var13591)
  #[1,2]
  var14553=tf.reshape(var14552, [1,2])
  #transpose: p = PermSwap; [32,32]
  #[32]
  var14554=tf.multiply(var13516, var14547)
  #[1,32]
  var14555=tf.reshape(var14554, [1,32])
  #transpose: p = PermSwap; [32,32]
  #transpose: p = PermSwap; [32,32]
  #[1]
  var14556=var13569[49:50]
  #[]
  var14557=tf.reshape(var14556, [])
  #[150]
  var14558=tf.gather(params=var13568, indices=var14557, batch_dims=0, axis=0)
  #[150]
  var14559=tf.multiply(var13567, var14558)
  #[1024]
  var14560=tf.gather(params=var14559, indices=var13553, batch_dims=0, axis=0)
  #[1024]
  var14561=tf.where(var13559, var14560, var13578)
  #[32,32]
  var14562=tf.reshape(var14561, [32,32])
  #transpose: p = PermSwap; [32,32]
  #[32,32]
  var14563=tf.transpose(var14562, perm=[1,0])
  #[32,32]
  var14564=tf.subtract(var14562, var14563)
  #[32,32]
  var14565=tf.linalg.expm(var14564)
  #[1,32]
  var14566=tf.matmul(var14555, var14565)
  #[32]
  var14567=tf.reshape(var14566, [32])
  #[32]
  var14568=tf.multiply(var13512, var14567)
  #[1,32]
  var14569=tf.reshape(var14568, [1,32])
  #[1,2]
  var14570=tf.matmul(var14569, var13588)
  #[2]
  var14571=tf.reshape(var14570, [2])
  #[2]
  var14572=tf.add(var14571, var13591)
  #[1,2]
  var14573=tf.reshape(var14572, [1,2])
  #[50,2]
  var14574=tf.concat([var13593
                     ,var13613
                     ,var13633
                     ,var13653
                     ,var13673
                     ,var13693
                     ,var13713
                     ,var13733
                     ,var13753
                     ,var13773
                     ,var13793
                     ,var13813
                     ,var13833
                     ,var13853
                     ,var13873
                     ,var13893
                     ,var13913
                     ,var13933
                     ,var13953
                     ,var13973
                     ,var13993
                     ,var14013
                     ,var14033
                     ,var14053
                     ,var14073
                     ,var14093
                     ,var14113
                     ,var14133
                     ,var14153
                     ,var14173
                     ,var14193
                     ,var14213
                     ,var14233
                     ,var14253
                     ,var14273
                     ,var14293
                     ,var14313
                     ,var14333
                     ,var14353
                     ,var14373
                     ,var14393
                     ,var14413
                     ,var14433
                     ,var14453
                     ,var14473
                     ,var14493
                     ,var14513
                     ,var14533
                     ,var14553
                     ,var14573],
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
  #[50]
  var14575=tf.argmax(var14574, axis=1, output_type=tf.int32)
  return {"pred":var14574,"y":var14575}
probePreds = {"function":probePreds_fn
             ,"batched":False
             ,"placeholders":{"x":{"shape":[50],"dtype":tf.int32}}}