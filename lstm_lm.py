
import tensorflow as tf
def mkModel():
  
  #[12,12]
  var12345=tf.random.uniform([12,12], minval=-5.0e-2, maxval=5.0e-2, dtype=tf.float32) # 0
  var12346=tf.Variable(name="embs", trainable=True, initial_value=var12345)
  #[160,160]
  var12347=tf.keras.initializers.orthogonal()(dtype=tf.float32, shape=[160,160]) # 1
  #[12,160]
  var12348=tf.random.uniform(
             [12,160], minval=-0.18677184, maxval=0.18677184, dtype=tf.float32) # 2
  #[172,160]
  var12349=tf.concat([var12347,var12348], axis=0)
  var12350=tf.Variable(name="w1_f_w", trainable=True, initial_value=var12349)
  #[]
  var12351=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[160]
  var12352=tf.broadcast_to(tf.reshape(var12351, [1]), [160])
  #[160]
  var12353=tf.reshape(var12352, [160])
  var12354=tf.Variable(name="w1_f_bias", trainable=True, initial_value=var12353)
  #[160,160]
  var12355=tf.keras.initializers.orthogonal()(dtype=tf.float32, shape=[160,160]) # 3
  #[12,160]
  var12356=tf.random.uniform(
             [12,160], minval=-0.18677184, maxval=0.18677184, dtype=tf.float32) # 4
  #[172,160]
  var12357=tf.concat([var12355,var12356], axis=0)
  var12358=tf.Variable(name="w1_i_w", trainable=True, initial_value=var12357)
  #[]
  var12359=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[160]
  var12360=tf.broadcast_to(tf.reshape(var12359, [1]), [160])
  #[160]
  var12361=tf.reshape(var12360, [160])
  var12362=tf.Variable(name="w1_i_bias", trainable=True, initial_value=var12361)
  #[160,160]
  var12363=tf.keras.initializers.orthogonal()(dtype=tf.float32, shape=[160,160]) # 5
  #[12,160]
  var12364=tf.random.uniform(
             [12,160], minval=-0.18677184, maxval=0.18677184, dtype=tf.float32) # 6
  #[172,160]
  var12365=tf.concat([var12363,var12364], axis=0)
  var12366=tf.Variable(name="w1_c_w", trainable=True, initial_value=var12365)
  var12367=tf.Variable(name="w1_c_bias", trainable=True, initial_value=var12361)
  #[160,160]
  var12368=tf.keras.initializers.orthogonal()(dtype=tf.float32, shape=[160,160]) # 7
  #[12,160]
  var12369=tf.random.uniform(
             [12,160], minval=-0.18677184, maxval=0.18677184, dtype=tf.float32) # 8
  #[172,160]
  var12370=tf.concat([var12368,var12369], axis=0)
  var12371=tf.Variable(name="w1_o_w", trainable=True, initial_value=var12370)
  var12372=tf.Variable(name="w1_o_bias", trainable=True, initial_value=var12361)
  #[160,12]
  var12373=tf.random.uniform(
             [160,12], minval=-0.18677184, maxval=0.18677184, dtype=tf.float32) # 12
  var12374=tf.Variable(name="dense_w", trainable=True, initial_value=var12373)
  #[12]
  var12375=tf.random.truncated_normal([12], stddev=0.1, dtype=tf.float32) # 13
  var12376=tf.Variable(name="dense_bias", trainable=True, initial_value=var12375)
  return {"batch_size":512
         ,"parameters":[var12346
                       ,var12350
                       ,var12354
                       ,var12358
                       ,var12362
                       ,var12366
                       ,var12367
                       ,var12371
                       ,var12372
                       ,var12374
                       ,var12376]
         ,"paramsdict":{"embs":var12346
                       ,"w1_f_w":var12350
                       ,"w1_f_bias":var12354
                       ,"w1_i_w":var12358
                       ,"w1_i_bias":var12362
                       ,"w1_c_w":var12366
                       ,"w1_c_bias":var12367
                       ,"w1_o_w":var12371
                       ,"w1_o_bias":var12372
                       ,"dense_w":var12374
                       ,"dense_bias":var12376}}
@tf.function
def runModel_fn(training_placeholder,
                embs,
                w1_f_w,
                w1_f_bias,
                w1_i_w,
                w1_i_bias,
                w1_c_w,
                w1_c_bias,
                w1_o_w,
                w1_o_bias,
                dense_w,
                dense_bias,
                x,
                y,
                weights):
  
  #[512,21]
  var12377=y
  #[]
  var12378=training_placeholder
  #[512,160]
  var12379=tf.random.uniform([512,160], minval=0.9, maxval=1.9, dtype=tf.float32) # 11
  #[512,160]
  var12380=tf.floor(var12379)
  #[]
  var12381=tf.constant(0.9, shape=[], dtype=tf.float32)
  #[160]
  var12382=tf.broadcast_to(tf.reshape(var12381, [1]), [160])
  #[160]
  var12383=tf.reshape(var12382, [160])
  #[512,160]
  var12384=tf.broadcast_to(tf.reshape(var12383, [1,160]), [512,160])
  #[512,160]
  var12385=tf.divide(var12380, var12384)
  #[]
  var12386=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[160]
  var12387=tf.broadcast_to(tf.reshape(var12386, [1]), [160])
  #[160]
  var12388=tf.reshape(var12387, [160])
  #[512,160]
  var12389=tf.broadcast_to(tf.reshape(var12388, [1,160]), [512,160])
  #[512,160]
  var12390=tf.cond(var12378, true_fn=lambda: var12385, false_fn=lambda: var12389)
  #[512,160]
  var12391=tf.random.uniform([512,160], minval=0.9, maxval=1.9, dtype=tf.float32) # 10
  #[512,160]
  var12392=tf.floor(var12391)
  #[512,160]
  var12393=tf.divide(var12392, var12384)
  #[512,160]
  var12394=tf.cond(var12378, true_fn=lambda: var12393, false_fn=lambda: var12389)
  #[]
  var12395=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[160]
  var12396=tf.broadcast_to(tf.reshape(var12395, [1]), [160])
  #[160]
  var12397=tf.reshape(var12396, [160])
  #[512,160]
  var12398=tf.broadcast_to(tf.reshape(var12397, [1,160]), [512,160])
  #[512,160]
  var12399=tf.multiply(var12394, var12398)
  #[512,12]
  var12400=tf.random.uniform([512,12], minval=0.9, maxval=1.9, dtype=tf.float32) # 9
  #[512,12]
  var12401=tf.floor(var12400)
  #[12]
  var12402=tf.broadcast_to(tf.reshape(var12381, [1]), [12])
  #[12]
  var12403=tf.reshape(var12402, [12])
  #[512,12]
  var12404=tf.broadcast_to(tf.reshape(var12403, [1,12]), [512,12])
  #[512,12]
  var12405=tf.divide(var12401, var12404)
  #[12]
  var12406=tf.broadcast_to(tf.reshape(var12386, [1]), [12])
  #[12]
  var12407=tf.reshape(var12406, [12])
  #[512,12]
  var12408=tf.broadcast_to(tf.reshape(var12407, [1,12]), [512,12])
  #[512,12]
  var12409=tf.cond(var12378, true_fn=lambda: var12405, false_fn=lambda: var12408)
  #[12,12]
  var12410=embs
  #[512,21]
  var12411=x
  #[512,1]
  var12412=var12411[:,0:1]
  #[512]
  var12413=tf.reshape(var12412, [512])
  #[512,12]
  var12414=tf.gather(params=var12410, indices=var12413, batch_dims=0, axis=0)
  #[512,12]
  var12415=tf.multiply(var12409, var12414)
  #[512,172]
  var12416=tf.concat([var12399,var12415], axis=1)
  #[512,172]
  var12417=tf.reshape(var12416, [512,172])
  #[172,160]
  var12418=w1_o_w
  #[512,160]
  var12419=tf.matmul(var12417, var12418)
  #[512,160]
  var12420=tf.reshape(var12419, [512,160])
  #[160]
  var12421=w1_o_bias
  #[512,160]
  var12422=tf.broadcast_to(tf.reshape(var12421, [1,160]), [512,160])
  #[512,160]
  var12423=tf.add(var12420, var12422)
  #[512,160]
  var12424=tf.sigmoid(var12423)
  #[172,160]
  var12425=w1_f_w
  #[512,160]
  var12426=tf.matmul(var12417, var12425)
  #[512,160]
  var12427=tf.reshape(var12426, [512,160])
  #[160]
  var12428=w1_f_bias
  #[512,160]
  var12429=tf.broadcast_to(tf.reshape(var12428, [1,160]), [512,160])
  #[512,160]
  var12430=tf.add(var12427, var12429)
  #[512,160]
  var12431=tf.sigmoid(var12430)
  #[512,160]
  var12432=tf.multiply(var12431, var12398)
  #[172,160]
  var12433=w1_i_w
  #[512,160]
  var12434=tf.matmul(var12417, var12433)
  #[512,160]
  var12435=tf.reshape(var12434, [512,160])
  #[160]
  var12436=w1_i_bias
  #[512,160]
  var12437=tf.broadcast_to(tf.reshape(var12436, [1,160]), [512,160])
  #[512,160]
  var12438=tf.add(var12435, var12437)
  #[512,160]
  var12439=tf.sigmoid(var12438)
  #[172,160]
  var12440=w1_c_w
  #[512,160]
  var12441=tf.matmul(var12417, var12440)
  #[512,160]
  var12442=tf.reshape(var12441, [512,160])
  #[160]
  var12443=w1_c_bias
  #[512,160]
  var12444=tf.broadcast_to(tf.reshape(var12443, [1,160]), [512,160])
  #[512,160]
  var12445=tf.add(var12442, var12444)
  #[512,160]
  var12446=tf.tanh(var12445)
  #[512,160]
  var12447=tf.multiply(var12439, var12446)
  #[512,160]
  var12448=tf.add(var12432, var12447)
  #[512,160]
  var12449=tf.tanh(var12448)
  #[512,160]
  var12450=tf.multiply(var12424, var12449)
  #[512,160]
  var12451=tf.multiply(var12390, var12450)
  #[512,160]
  var12452=tf.reshape(var12451, [512,160])
  #[160,12]
  var12453=dense_w
  #[512,12]
  var12454=tf.matmul(var12452, var12453)
  #[512,12]
  var12455=tf.reshape(var12454, [512,12])
  #[12]
  var12456=dense_bias
  #[512,12]
  var12457=tf.broadcast_to(tf.reshape(var12456, [1,12]), [512,12])
  #[512,12]
  var12458=tf.add(var12455, var12457)
  #[512,1,12]
  var12459=tf.reshape(var12458, [512,1,12])
  #[512,160]
  var12460=tf.multiply(var12394, var12450)
  #[512,1]
  var12461=var12411[:,1:2]
  #[512]
  var12462=tf.reshape(var12461, [512])
  #[512,12]
  var12463=tf.gather(params=var12410, indices=var12462, batch_dims=0, axis=0)
  #[512,12]
  var12464=tf.multiply(var12409, var12463)
  #[512,172]
  var12465=tf.concat([var12460,var12464], axis=1)
  #[512,172]
  var12466=tf.reshape(var12465, [512,172])
  #[512,160]
  var12467=tf.matmul(var12466, var12418)
  #[512,160]
  var12468=tf.reshape(var12467, [512,160])
  #[512,160]
  var12469=tf.add(var12468, var12422)
  #[512,160]
  var12470=tf.sigmoid(var12469)
  #[512,160]
  var12471=tf.matmul(var12466, var12425)
  #[512,160]
  var12472=tf.reshape(var12471, [512,160])
  #[512,160]
  var12473=tf.add(var12472, var12429)
  #[512,160]
  var12474=tf.sigmoid(var12473)
  #[512,160]
  var12475=tf.multiply(var12474, var12448)
  #[512,160]
  var12476=tf.matmul(var12466, var12433)
  #[512,160]
  var12477=tf.reshape(var12476, [512,160])
  #[512,160]
  var12478=tf.add(var12477, var12437)
  #[512,160]
  var12479=tf.sigmoid(var12478)
  #[512,160]
  var12480=tf.matmul(var12466, var12440)
  #[512,160]
  var12481=tf.reshape(var12480, [512,160])
  #[512,160]
  var12482=tf.add(var12481, var12444)
  #[512,160]
  var12483=tf.tanh(var12482)
  #[512,160]
  var12484=tf.multiply(var12479, var12483)
  #[512,160]
  var12485=tf.add(var12475, var12484)
  #[512,160]
  var12486=tf.tanh(var12485)
  #[512,160]
  var12487=tf.multiply(var12470, var12486)
  #[512,160]
  var12488=tf.multiply(var12390, var12487)
  #[512,160]
  var12489=tf.reshape(var12488, [512,160])
  #[512,12]
  var12490=tf.matmul(var12489, var12453)
  #[512,12]
  var12491=tf.reshape(var12490, [512,12])
  #[512,12]
  var12492=tf.add(var12491, var12457)
  #[512,1,12]
  var12493=tf.reshape(var12492, [512,1,12])
  #[512,160]
  var12494=tf.multiply(var12394, var12487)
  #[512,1]
  var12495=var12411[:,2:3]
  #[512]
  var12496=tf.reshape(var12495, [512])
  #[512,12]
  var12497=tf.gather(params=var12410, indices=var12496, batch_dims=0, axis=0)
  #[512,12]
  var12498=tf.multiply(var12409, var12497)
  #[512,172]
  var12499=tf.concat([var12494,var12498], axis=1)
  #[512,172]
  var12500=tf.reshape(var12499, [512,172])
  #[512,160]
  var12501=tf.matmul(var12500, var12418)
  #[512,160]
  var12502=tf.reshape(var12501, [512,160])
  #[512,160]
  var12503=tf.add(var12502, var12422)
  #[512,160]
  var12504=tf.sigmoid(var12503)
  #[512,160]
  var12505=tf.matmul(var12500, var12425)
  #[512,160]
  var12506=tf.reshape(var12505, [512,160])
  #[512,160]
  var12507=tf.add(var12506, var12429)
  #[512,160]
  var12508=tf.sigmoid(var12507)
  #[512,160]
  var12509=tf.multiply(var12508, var12485)
  #[512,160]
  var12510=tf.matmul(var12500, var12433)
  #[512,160]
  var12511=tf.reshape(var12510, [512,160])
  #[512,160]
  var12512=tf.add(var12511, var12437)
  #[512,160]
  var12513=tf.sigmoid(var12512)
  #[512,160]
  var12514=tf.matmul(var12500, var12440)
  #[512,160]
  var12515=tf.reshape(var12514, [512,160])
  #[512,160]
  var12516=tf.add(var12515, var12444)
  #[512,160]
  var12517=tf.tanh(var12516)
  #[512,160]
  var12518=tf.multiply(var12513, var12517)
  #[512,160]
  var12519=tf.add(var12509, var12518)
  #[512,160]
  var12520=tf.tanh(var12519)
  #[512,160]
  var12521=tf.multiply(var12504, var12520)
  #[512,160]
  var12522=tf.multiply(var12390, var12521)
  #[512,160]
  var12523=tf.reshape(var12522, [512,160])
  #[512,12]
  var12524=tf.matmul(var12523, var12453)
  #[512,12]
  var12525=tf.reshape(var12524, [512,12])
  #[512,12]
  var12526=tf.add(var12525, var12457)
  #[512,1,12]
  var12527=tf.reshape(var12526, [512,1,12])
  #[512,160]
  var12528=tf.multiply(var12394, var12521)
  #[512,1]
  var12529=var12411[:,3:4]
  #[512]
  var12530=tf.reshape(var12529, [512])
  #[512,12]
  var12531=tf.gather(params=var12410, indices=var12530, batch_dims=0, axis=0)
  #[512,12]
  var12532=tf.multiply(var12409, var12531)
  #[512,172]
  var12533=tf.concat([var12528,var12532], axis=1)
  #[512,172]
  var12534=tf.reshape(var12533, [512,172])
  #[512,160]
  var12535=tf.matmul(var12534, var12418)
  #[512,160]
  var12536=tf.reshape(var12535, [512,160])
  #[512,160]
  var12537=tf.add(var12536, var12422)
  #[512,160]
  var12538=tf.sigmoid(var12537)
  #[512,160]
  var12539=tf.matmul(var12534, var12425)
  #[512,160]
  var12540=tf.reshape(var12539, [512,160])
  #[512,160]
  var12541=tf.add(var12540, var12429)
  #[512,160]
  var12542=tf.sigmoid(var12541)
  #[512,160]
  var12543=tf.multiply(var12542, var12519)
  #[512,160]
  var12544=tf.matmul(var12534, var12433)
  #[512,160]
  var12545=tf.reshape(var12544, [512,160])
  #[512,160]
  var12546=tf.add(var12545, var12437)
  #[512,160]
  var12547=tf.sigmoid(var12546)
  #[512,160]
  var12548=tf.matmul(var12534, var12440)
  #[512,160]
  var12549=tf.reshape(var12548, [512,160])
  #[512,160]
  var12550=tf.add(var12549, var12444)
  #[512,160]
  var12551=tf.tanh(var12550)
  #[512,160]
  var12552=tf.multiply(var12547, var12551)
  #[512,160]
  var12553=tf.add(var12543, var12552)
  #[512,160]
  var12554=tf.tanh(var12553)
  #[512,160]
  var12555=tf.multiply(var12538, var12554)
  #[512,160]
  var12556=tf.multiply(var12390, var12555)
  #[512,160]
  var12557=tf.reshape(var12556, [512,160])
  #[512,12]
  var12558=tf.matmul(var12557, var12453)
  #[512,12]
  var12559=tf.reshape(var12558, [512,12])
  #[512,12]
  var12560=tf.add(var12559, var12457)
  #[512,1,12]
  var12561=tf.reshape(var12560, [512,1,12])
  #[512,160]
  var12562=tf.multiply(var12394, var12555)
  #[512,1]
  var12563=var12411[:,4:5]
  #[512]
  var12564=tf.reshape(var12563, [512])
  #[512,12]
  var12565=tf.gather(params=var12410, indices=var12564, batch_dims=0, axis=0)
  #[512,12]
  var12566=tf.multiply(var12409, var12565)
  #[512,172]
  var12567=tf.concat([var12562,var12566], axis=1)
  #[512,172]
  var12568=tf.reshape(var12567, [512,172])
  #[512,160]
  var12569=tf.matmul(var12568, var12418)
  #[512,160]
  var12570=tf.reshape(var12569, [512,160])
  #[512,160]
  var12571=tf.add(var12570, var12422)
  #[512,160]
  var12572=tf.sigmoid(var12571)
  #[512,160]
  var12573=tf.matmul(var12568, var12425)
  #[512,160]
  var12574=tf.reshape(var12573, [512,160])
  #[512,160]
  var12575=tf.add(var12574, var12429)
  #[512,160]
  var12576=tf.sigmoid(var12575)
  #[512,160]
  var12577=tf.multiply(var12576, var12553)
  #[512,160]
  var12578=tf.matmul(var12568, var12433)
  #[512,160]
  var12579=tf.reshape(var12578, [512,160])
  #[512,160]
  var12580=tf.add(var12579, var12437)
  #[512,160]
  var12581=tf.sigmoid(var12580)
  #[512,160]
  var12582=tf.matmul(var12568, var12440)
  #[512,160]
  var12583=tf.reshape(var12582, [512,160])
  #[512,160]
  var12584=tf.add(var12583, var12444)
  #[512,160]
  var12585=tf.tanh(var12584)
  #[512,160]
  var12586=tf.multiply(var12581, var12585)
  #[512,160]
  var12587=tf.add(var12577, var12586)
  #[512,160]
  var12588=tf.tanh(var12587)
  #[512,160]
  var12589=tf.multiply(var12572, var12588)
  #[512,160]
  var12590=tf.multiply(var12390, var12589)
  #[512,160]
  var12591=tf.reshape(var12590, [512,160])
  #[512,12]
  var12592=tf.matmul(var12591, var12453)
  #[512,12]
  var12593=tf.reshape(var12592, [512,12])
  #[512,12]
  var12594=tf.add(var12593, var12457)
  #[512,1,12]
  var12595=tf.reshape(var12594, [512,1,12])
  #[512,160]
  var12596=tf.multiply(var12394, var12589)
  #[512,1]
  var12597=var12411[:,5:6]
  #[512]
  var12598=tf.reshape(var12597, [512])
  #[512,12]
  var12599=tf.gather(params=var12410, indices=var12598, batch_dims=0, axis=0)
  #[512,12]
  var12600=tf.multiply(var12409, var12599)
  #[512,172]
  var12601=tf.concat([var12596,var12600], axis=1)
  #[512,172]
  var12602=tf.reshape(var12601, [512,172])
  #[512,160]
  var12603=tf.matmul(var12602, var12418)
  #[512,160]
  var12604=tf.reshape(var12603, [512,160])
  #[512,160]
  var12605=tf.add(var12604, var12422)
  #[512,160]
  var12606=tf.sigmoid(var12605)
  #[512,160]
  var12607=tf.matmul(var12602, var12425)
  #[512,160]
  var12608=tf.reshape(var12607, [512,160])
  #[512,160]
  var12609=tf.add(var12608, var12429)
  #[512,160]
  var12610=tf.sigmoid(var12609)
  #[512,160]
  var12611=tf.multiply(var12610, var12587)
  #[512,160]
  var12612=tf.matmul(var12602, var12433)
  #[512,160]
  var12613=tf.reshape(var12612, [512,160])
  #[512,160]
  var12614=tf.add(var12613, var12437)
  #[512,160]
  var12615=tf.sigmoid(var12614)
  #[512,160]
  var12616=tf.matmul(var12602, var12440)
  #[512,160]
  var12617=tf.reshape(var12616, [512,160])
  #[512,160]
  var12618=tf.add(var12617, var12444)
  #[512,160]
  var12619=tf.tanh(var12618)
  #[512,160]
  var12620=tf.multiply(var12615, var12619)
  #[512,160]
  var12621=tf.add(var12611, var12620)
  #[512,160]
  var12622=tf.tanh(var12621)
  #[512,160]
  var12623=tf.multiply(var12606, var12622)
  #[512,160]
  var12624=tf.multiply(var12390, var12623)
  #[512,160]
  var12625=tf.reshape(var12624, [512,160])
  #[512,12]
  var12626=tf.matmul(var12625, var12453)
  #[512,12]
  var12627=tf.reshape(var12626, [512,12])
  #[512,12]
  var12628=tf.add(var12627, var12457)
  #[512,1,12]
  var12629=tf.reshape(var12628, [512,1,12])
  #[512,160]
  var12630=tf.multiply(var12394, var12623)
  #[512,1]
  var12631=var12411[:,6:7]
  #[512]
  var12632=tf.reshape(var12631, [512])
  #[512,12]
  var12633=tf.gather(params=var12410, indices=var12632, batch_dims=0, axis=0)
  #[512,12]
  var12634=tf.multiply(var12409, var12633)
  #[512,172]
  var12635=tf.concat([var12630,var12634], axis=1)
  #[512,172]
  var12636=tf.reshape(var12635, [512,172])
  #[512,160]
  var12637=tf.matmul(var12636, var12418)
  #[512,160]
  var12638=tf.reshape(var12637, [512,160])
  #[512,160]
  var12639=tf.add(var12638, var12422)
  #[512,160]
  var12640=tf.sigmoid(var12639)
  #[512,160]
  var12641=tf.matmul(var12636, var12425)
  #[512,160]
  var12642=tf.reshape(var12641, [512,160])
  #[512,160]
  var12643=tf.add(var12642, var12429)
  #[512,160]
  var12644=tf.sigmoid(var12643)
  #[512,160]
  var12645=tf.multiply(var12644, var12621)
  #[512,160]
  var12646=tf.matmul(var12636, var12433)
  #[512,160]
  var12647=tf.reshape(var12646, [512,160])
  #[512,160]
  var12648=tf.add(var12647, var12437)
  #[512,160]
  var12649=tf.sigmoid(var12648)
  #[512,160]
  var12650=tf.matmul(var12636, var12440)
  #[512,160]
  var12651=tf.reshape(var12650, [512,160])
  #[512,160]
  var12652=tf.add(var12651, var12444)
  #[512,160]
  var12653=tf.tanh(var12652)
  #[512,160]
  var12654=tf.multiply(var12649, var12653)
  #[512,160]
  var12655=tf.add(var12645, var12654)
  #[512,160]
  var12656=tf.tanh(var12655)
  #[512,160]
  var12657=tf.multiply(var12640, var12656)
  #[512,160]
  var12658=tf.multiply(var12390, var12657)
  #[512,160]
  var12659=tf.reshape(var12658, [512,160])
  #[512,12]
  var12660=tf.matmul(var12659, var12453)
  #[512,12]
  var12661=tf.reshape(var12660, [512,12])
  #[512,12]
  var12662=tf.add(var12661, var12457)
  #[512,1,12]
  var12663=tf.reshape(var12662, [512,1,12])
  #[512,160]
  var12664=tf.multiply(var12394, var12657)
  #[512,1]
  var12665=var12411[:,7:8]
  #[512]
  var12666=tf.reshape(var12665, [512])
  #[512,12]
  var12667=tf.gather(params=var12410, indices=var12666, batch_dims=0, axis=0)
  #[512,12]
  var12668=tf.multiply(var12409, var12667)
  #[512,172]
  var12669=tf.concat([var12664,var12668], axis=1)
  #[512,172]
  var12670=tf.reshape(var12669, [512,172])
  #[512,160]
  var12671=tf.matmul(var12670, var12418)
  #[512,160]
  var12672=tf.reshape(var12671, [512,160])
  #[512,160]
  var12673=tf.add(var12672, var12422)
  #[512,160]
  var12674=tf.sigmoid(var12673)
  #[512,160]
  var12675=tf.matmul(var12670, var12425)
  #[512,160]
  var12676=tf.reshape(var12675, [512,160])
  #[512,160]
  var12677=tf.add(var12676, var12429)
  #[512,160]
  var12678=tf.sigmoid(var12677)
  #[512,160]
  var12679=tf.multiply(var12678, var12655)
  #[512,160]
  var12680=tf.matmul(var12670, var12433)
  #[512,160]
  var12681=tf.reshape(var12680, [512,160])
  #[512,160]
  var12682=tf.add(var12681, var12437)
  #[512,160]
  var12683=tf.sigmoid(var12682)
  #[512,160]
  var12684=tf.matmul(var12670, var12440)
  #[512,160]
  var12685=tf.reshape(var12684, [512,160])
  #[512,160]
  var12686=tf.add(var12685, var12444)
  #[512,160]
  var12687=tf.tanh(var12686)
  #[512,160]
  var12688=tf.multiply(var12683, var12687)
  #[512,160]
  var12689=tf.add(var12679, var12688)
  #[512,160]
  var12690=tf.tanh(var12689)
  #[512,160]
  var12691=tf.multiply(var12674, var12690)
  #[512,160]
  var12692=tf.multiply(var12390, var12691)
  #[512,160]
  var12693=tf.reshape(var12692, [512,160])
  #[512,12]
  var12694=tf.matmul(var12693, var12453)
  #[512,12]
  var12695=tf.reshape(var12694, [512,12])
  #[512,12]
  var12696=tf.add(var12695, var12457)
  #[512,1,12]
  var12697=tf.reshape(var12696, [512,1,12])
  #[512,160]
  var12698=tf.multiply(var12394, var12691)
  #[512,1]
  var12699=var12411[:,8:9]
  #[512]
  var12700=tf.reshape(var12699, [512])
  #[512,12]
  var12701=tf.gather(params=var12410, indices=var12700, batch_dims=0, axis=0)
  #[512,12]
  var12702=tf.multiply(var12409, var12701)
  #[512,172]
  var12703=tf.concat([var12698,var12702], axis=1)
  #[512,172]
  var12704=tf.reshape(var12703, [512,172])
  #[512,160]
  var12705=tf.matmul(var12704, var12418)
  #[512,160]
  var12706=tf.reshape(var12705, [512,160])
  #[512,160]
  var12707=tf.add(var12706, var12422)
  #[512,160]
  var12708=tf.sigmoid(var12707)
  #[512,160]
  var12709=tf.matmul(var12704, var12425)
  #[512,160]
  var12710=tf.reshape(var12709, [512,160])
  #[512,160]
  var12711=tf.add(var12710, var12429)
  #[512,160]
  var12712=tf.sigmoid(var12711)
  #[512,160]
  var12713=tf.multiply(var12712, var12689)
  #[512,160]
  var12714=tf.matmul(var12704, var12433)
  #[512,160]
  var12715=tf.reshape(var12714, [512,160])
  #[512,160]
  var12716=tf.add(var12715, var12437)
  #[512,160]
  var12717=tf.sigmoid(var12716)
  #[512,160]
  var12718=tf.matmul(var12704, var12440)
  #[512,160]
  var12719=tf.reshape(var12718, [512,160])
  #[512,160]
  var12720=tf.add(var12719, var12444)
  #[512,160]
  var12721=tf.tanh(var12720)
  #[512,160]
  var12722=tf.multiply(var12717, var12721)
  #[512,160]
  var12723=tf.add(var12713, var12722)
  #[512,160]
  var12724=tf.tanh(var12723)
  #[512,160]
  var12725=tf.multiply(var12708, var12724)
  #[512,160]
  var12726=tf.multiply(var12390, var12725)
  #[512,160]
  var12727=tf.reshape(var12726, [512,160])
  #[512,12]
  var12728=tf.matmul(var12727, var12453)
  #[512,12]
  var12729=tf.reshape(var12728, [512,12])
  #[512,12]
  var12730=tf.add(var12729, var12457)
  #[512,1,12]
  var12731=tf.reshape(var12730, [512,1,12])
  #[512,160]
  var12732=tf.multiply(var12394, var12725)
  #[512,1]
  var12733=var12411[:,9:10]
  #[512]
  var12734=tf.reshape(var12733, [512])
  #[512,12]
  var12735=tf.gather(params=var12410, indices=var12734, batch_dims=0, axis=0)
  #[512,12]
  var12736=tf.multiply(var12409, var12735)
  #[512,172]
  var12737=tf.concat([var12732,var12736], axis=1)
  #[512,172]
  var12738=tf.reshape(var12737, [512,172])
  #[512,160]
  var12739=tf.matmul(var12738, var12418)
  #[512,160]
  var12740=tf.reshape(var12739, [512,160])
  #[512,160]
  var12741=tf.add(var12740, var12422)
  #[512,160]
  var12742=tf.sigmoid(var12741)
  #[512,160]
  var12743=tf.matmul(var12738, var12425)
  #[512,160]
  var12744=tf.reshape(var12743, [512,160])
  #[512,160]
  var12745=tf.add(var12744, var12429)
  #[512,160]
  var12746=tf.sigmoid(var12745)
  #[512,160]
  var12747=tf.multiply(var12746, var12723)
  #[512,160]
  var12748=tf.matmul(var12738, var12433)
  #[512,160]
  var12749=tf.reshape(var12748, [512,160])
  #[512,160]
  var12750=tf.add(var12749, var12437)
  #[512,160]
  var12751=tf.sigmoid(var12750)
  #[512,160]
  var12752=tf.matmul(var12738, var12440)
  #[512,160]
  var12753=tf.reshape(var12752, [512,160])
  #[512,160]
  var12754=tf.add(var12753, var12444)
  #[512,160]
  var12755=tf.tanh(var12754)
  #[512,160]
  var12756=tf.multiply(var12751, var12755)
  #[512,160]
  var12757=tf.add(var12747, var12756)
  #[512,160]
  var12758=tf.tanh(var12757)
  #[512,160]
  var12759=tf.multiply(var12742, var12758)
  #[512,160]
  var12760=tf.multiply(var12390, var12759)
  #[512,160]
  var12761=tf.reshape(var12760, [512,160])
  #[512,12]
  var12762=tf.matmul(var12761, var12453)
  #[512,12]
  var12763=tf.reshape(var12762, [512,12])
  #[512,12]
  var12764=tf.add(var12763, var12457)
  #[512,1,12]
  var12765=tf.reshape(var12764, [512,1,12])
  #[512,160]
  var12766=tf.multiply(var12394, var12759)
  #[512,1]
  var12767=var12411[:,10:11]
  #[512]
  var12768=tf.reshape(var12767, [512])
  #[512,12]
  var12769=tf.gather(params=var12410, indices=var12768, batch_dims=0, axis=0)
  #[512,12]
  var12770=tf.multiply(var12409, var12769)
  #[512,172]
  var12771=tf.concat([var12766,var12770], axis=1)
  #[512,172]
  var12772=tf.reshape(var12771, [512,172])
  #[512,160]
  var12773=tf.matmul(var12772, var12418)
  #[512,160]
  var12774=tf.reshape(var12773, [512,160])
  #[512,160]
  var12775=tf.add(var12774, var12422)
  #[512,160]
  var12776=tf.sigmoid(var12775)
  #[512,160]
  var12777=tf.matmul(var12772, var12425)
  #[512,160]
  var12778=tf.reshape(var12777, [512,160])
  #[512,160]
  var12779=tf.add(var12778, var12429)
  #[512,160]
  var12780=tf.sigmoid(var12779)
  #[512,160]
  var12781=tf.multiply(var12780, var12757)
  #[512,160]
  var12782=tf.matmul(var12772, var12433)
  #[512,160]
  var12783=tf.reshape(var12782, [512,160])
  #[512,160]
  var12784=tf.add(var12783, var12437)
  #[512,160]
  var12785=tf.sigmoid(var12784)
  #[512,160]
  var12786=tf.matmul(var12772, var12440)
  #[512,160]
  var12787=tf.reshape(var12786, [512,160])
  #[512,160]
  var12788=tf.add(var12787, var12444)
  #[512,160]
  var12789=tf.tanh(var12788)
  #[512,160]
  var12790=tf.multiply(var12785, var12789)
  #[512,160]
  var12791=tf.add(var12781, var12790)
  #[512,160]
  var12792=tf.tanh(var12791)
  #[512,160]
  var12793=tf.multiply(var12776, var12792)
  #[512,160]
  var12794=tf.multiply(var12390, var12793)
  #[512,160]
  var12795=tf.reshape(var12794, [512,160])
  #[512,12]
  var12796=tf.matmul(var12795, var12453)
  #[512,12]
  var12797=tf.reshape(var12796, [512,12])
  #[512,12]
  var12798=tf.add(var12797, var12457)
  #[512,1,12]
  var12799=tf.reshape(var12798, [512,1,12])
  #[512,160]
  var12800=tf.multiply(var12394, var12793)
  #[512,1]
  var12801=var12411[:,11:12]
  #[512]
  var12802=tf.reshape(var12801, [512])
  #[512,12]
  var12803=tf.gather(params=var12410, indices=var12802, batch_dims=0, axis=0)
  #[512,12]
  var12804=tf.multiply(var12409, var12803)
  #[512,172]
  var12805=tf.concat([var12800,var12804], axis=1)
  #[512,172]
  var12806=tf.reshape(var12805, [512,172])
  #[512,160]
  var12807=tf.matmul(var12806, var12418)
  #[512,160]
  var12808=tf.reshape(var12807, [512,160])
  #[512,160]
  var12809=tf.add(var12808, var12422)
  #[512,160]
  var12810=tf.sigmoid(var12809)
  #[512,160]
  var12811=tf.matmul(var12806, var12425)
  #[512,160]
  var12812=tf.reshape(var12811, [512,160])
  #[512,160]
  var12813=tf.add(var12812, var12429)
  #[512,160]
  var12814=tf.sigmoid(var12813)
  #[512,160]
  var12815=tf.multiply(var12814, var12791)
  #[512,160]
  var12816=tf.matmul(var12806, var12433)
  #[512,160]
  var12817=tf.reshape(var12816, [512,160])
  #[512,160]
  var12818=tf.add(var12817, var12437)
  #[512,160]
  var12819=tf.sigmoid(var12818)
  #[512,160]
  var12820=tf.matmul(var12806, var12440)
  #[512,160]
  var12821=tf.reshape(var12820, [512,160])
  #[512,160]
  var12822=tf.add(var12821, var12444)
  #[512,160]
  var12823=tf.tanh(var12822)
  #[512,160]
  var12824=tf.multiply(var12819, var12823)
  #[512,160]
  var12825=tf.add(var12815, var12824)
  #[512,160]
  var12826=tf.tanh(var12825)
  #[512,160]
  var12827=tf.multiply(var12810, var12826)
  #[512,160]
  var12828=tf.multiply(var12390, var12827)
  #[512,160]
  var12829=tf.reshape(var12828, [512,160])
  #[512,12]
  var12830=tf.matmul(var12829, var12453)
  #[512,12]
  var12831=tf.reshape(var12830, [512,12])
  #[512,12]
  var12832=tf.add(var12831, var12457)
  #[512,1,12]
  var12833=tf.reshape(var12832, [512,1,12])
  #[512,160]
  var12834=tf.multiply(var12394, var12827)
  #[512,1]
  var12835=var12411[:,12:13]
  #[512]
  var12836=tf.reshape(var12835, [512])
  #[512,12]
  var12837=tf.gather(params=var12410, indices=var12836, batch_dims=0, axis=0)
  #[512,12]
  var12838=tf.multiply(var12409, var12837)
  #[512,172]
  var12839=tf.concat([var12834,var12838], axis=1)
  #[512,172]
  var12840=tf.reshape(var12839, [512,172])
  #[512,160]
  var12841=tf.matmul(var12840, var12418)
  #[512,160]
  var12842=tf.reshape(var12841, [512,160])
  #[512,160]
  var12843=tf.add(var12842, var12422)
  #[512,160]
  var12844=tf.sigmoid(var12843)
  #[512,160]
  var12845=tf.matmul(var12840, var12425)
  #[512,160]
  var12846=tf.reshape(var12845, [512,160])
  #[512,160]
  var12847=tf.add(var12846, var12429)
  #[512,160]
  var12848=tf.sigmoid(var12847)
  #[512,160]
  var12849=tf.multiply(var12848, var12825)
  #[512,160]
  var12850=tf.matmul(var12840, var12433)
  #[512,160]
  var12851=tf.reshape(var12850, [512,160])
  #[512,160]
  var12852=tf.add(var12851, var12437)
  #[512,160]
  var12853=tf.sigmoid(var12852)
  #[512,160]
  var12854=tf.matmul(var12840, var12440)
  #[512,160]
  var12855=tf.reshape(var12854, [512,160])
  #[512,160]
  var12856=tf.add(var12855, var12444)
  #[512,160]
  var12857=tf.tanh(var12856)
  #[512,160]
  var12858=tf.multiply(var12853, var12857)
  #[512,160]
  var12859=tf.add(var12849, var12858)
  #[512,160]
  var12860=tf.tanh(var12859)
  #[512,160]
  var12861=tf.multiply(var12844, var12860)
  #[512,160]
  var12862=tf.multiply(var12390, var12861)
  #[512,160]
  var12863=tf.reshape(var12862, [512,160])
  #[512,12]
  var12864=tf.matmul(var12863, var12453)
  #[512,12]
  var12865=tf.reshape(var12864, [512,12])
  #[512,12]
  var12866=tf.add(var12865, var12457)
  #[512,1,12]
  var12867=tf.reshape(var12866, [512,1,12])
  #[512,160]
  var12868=tf.multiply(var12394, var12861)
  #[512,1]
  var12869=var12411[:,13:14]
  #[512]
  var12870=tf.reshape(var12869, [512])
  #[512,12]
  var12871=tf.gather(params=var12410, indices=var12870, batch_dims=0, axis=0)
  #[512,12]
  var12872=tf.multiply(var12409, var12871)
  #[512,172]
  var12873=tf.concat([var12868,var12872], axis=1)
  #[512,172]
  var12874=tf.reshape(var12873, [512,172])
  #[512,160]
  var12875=tf.matmul(var12874, var12418)
  #[512,160]
  var12876=tf.reshape(var12875, [512,160])
  #[512,160]
  var12877=tf.add(var12876, var12422)
  #[512,160]
  var12878=tf.sigmoid(var12877)
  #[512,160]
  var12879=tf.matmul(var12874, var12425)
  #[512,160]
  var12880=tf.reshape(var12879, [512,160])
  #[512,160]
  var12881=tf.add(var12880, var12429)
  #[512,160]
  var12882=tf.sigmoid(var12881)
  #[512,160]
  var12883=tf.multiply(var12882, var12859)
  #[512,160]
  var12884=tf.matmul(var12874, var12433)
  #[512,160]
  var12885=tf.reshape(var12884, [512,160])
  #[512,160]
  var12886=tf.add(var12885, var12437)
  #[512,160]
  var12887=tf.sigmoid(var12886)
  #[512,160]
  var12888=tf.matmul(var12874, var12440)
  #[512,160]
  var12889=tf.reshape(var12888, [512,160])
  #[512,160]
  var12890=tf.add(var12889, var12444)
  #[512,160]
  var12891=tf.tanh(var12890)
  #[512,160]
  var12892=tf.multiply(var12887, var12891)
  #[512,160]
  var12893=tf.add(var12883, var12892)
  #[512,160]
  var12894=tf.tanh(var12893)
  #[512,160]
  var12895=tf.multiply(var12878, var12894)
  #[512,160]
  var12896=tf.multiply(var12390, var12895)
  #[512,160]
  var12897=tf.reshape(var12896, [512,160])
  #[512,12]
  var12898=tf.matmul(var12897, var12453)
  #[512,12]
  var12899=tf.reshape(var12898, [512,12])
  #[512,12]
  var12900=tf.add(var12899, var12457)
  #[512,1,12]
  var12901=tf.reshape(var12900, [512,1,12])
  #[512,160]
  var12902=tf.multiply(var12394, var12895)
  #[512,1]
  var12903=var12411[:,14:15]
  #[512]
  var12904=tf.reshape(var12903, [512])
  #[512,12]
  var12905=tf.gather(params=var12410, indices=var12904, batch_dims=0, axis=0)
  #[512,12]
  var12906=tf.multiply(var12409, var12905)
  #[512,172]
  var12907=tf.concat([var12902,var12906], axis=1)
  #[512,172]
  var12908=tf.reshape(var12907, [512,172])
  #[512,160]
  var12909=tf.matmul(var12908, var12418)
  #[512,160]
  var12910=tf.reshape(var12909, [512,160])
  #[512,160]
  var12911=tf.add(var12910, var12422)
  #[512,160]
  var12912=tf.sigmoid(var12911)
  #[512,160]
  var12913=tf.matmul(var12908, var12425)
  #[512,160]
  var12914=tf.reshape(var12913, [512,160])
  #[512,160]
  var12915=tf.add(var12914, var12429)
  #[512,160]
  var12916=tf.sigmoid(var12915)
  #[512,160]
  var12917=tf.multiply(var12916, var12893)
  #[512,160]
  var12918=tf.matmul(var12908, var12433)
  #[512,160]
  var12919=tf.reshape(var12918, [512,160])
  #[512,160]
  var12920=tf.add(var12919, var12437)
  #[512,160]
  var12921=tf.sigmoid(var12920)
  #[512,160]
  var12922=tf.matmul(var12908, var12440)
  #[512,160]
  var12923=tf.reshape(var12922, [512,160])
  #[512,160]
  var12924=tf.add(var12923, var12444)
  #[512,160]
  var12925=tf.tanh(var12924)
  #[512,160]
  var12926=tf.multiply(var12921, var12925)
  #[512,160]
  var12927=tf.add(var12917, var12926)
  #[512,160]
  var12928=tf.tanh(var12927)
  #[512,160]
  var12929=tf.multiply(var12912, var12928)
  #[512,160]
  var12930=tf.multiply(var12390, var12929)
  #[512,160]
  var12931=tf.reshape(var12930, [512,160])
  #[512,12]
  var12932=tf.matmul(var12931, var12453)
  #[512,12]
  var12933=tf.reshape(var12932, [512,12])
  #[512,12]
  var12934=tf.add(var12933, var12457)
  #[512,1,12]
  var12935=tf.reshape(var12934, [512,1,12])
  #[512,160]
  var12936=tf.multiply(var12394, var12929)
  #[512,1]
  var12937=var12411[:,15:16]
  #[512]
  var12938=tf.reshape(var12937, [512])
  #[512,12]
  var12939=tf.gather(params=var12410, indices=var12938, batch_dims=0, axis=0)
  #[512,12]
  var12940=tf.multiply(var12409, var12939)
  #[512,172]
  var12941=tf.concat([var12936,var12940], axis=1)
  #[512,172]
  var12942=tf.reshape(var12941, [512,172])
  #[512,160]
  var12943=tf.matmul(var12942, var12418)
  #[512,160]
  var12944=tf.reshape(var12943, [512,160])
  #[512,160]
  var12945=tf.add(var12944, var12422)
  #[512,160]
  var12946=tf.sigmoid(var12945)
  #[512,160]
  var12947=tf.matmul(var12942, var12425)
  #[512,160]
  var12948=tf.reshape(var12947, [512,160])
  #[512,160]
  var12949=tf.add(var12948, var12429)
  #[512,160]
  var12950=tf.sigmoid(var12949)
  #[512,160]
  var12951=tf.multiply(var12950, var12927)
  #[512,160]
  var12952=tf.matmul(var12942, var12433)
  #[512,160]
  var12953=tf.reshape(var12952, [512,160])
  #[512,160]
  var12954=tf.add(var12953, var12437)
  #[512,160]
  var12955=tf.sigmoid(var12954)
  #[512,160]
  var12956=tf.matmul(var12942, var12440)
  #[512,160]
  var12957=tf.reshape(var12956, [512,160])
  #[512,160]
  var12958=tf.add(var12957, var12444)
  #[512,160]
  var12959=tf.tanh(var12958)
  #[512,160]
  var12960=tf.multiply(var12955, var12959)
  #[512,160]
  var12961=tf.add(var12951, var12960)
  #[512,160]
  var12962=tf.tanh(var12961)
  #[512,160]
  var12963=tf.multiply(var12946, var12962)
  #[512,160]
  var12964=tf.multiply(var12390, var12963)
  #[512,160]
  var12965=tf.reshape(var12964, [512,160])
  #[512,12]
  var12966=tf.matmul(var12965, var12453)
  #[512,12]
  var12967=tf.reshape(var12966, [512,12])
  #[512,12]
  var12968=tf.add(var12967, var12457)
  #[512,1,12]
  var12969=tf.reshape(var12968, [512,1,12])
  #[512,160]
  var12970=tf.multiply(var12394, var12963)
  #[512,1]
  var12971=var12411[:,16:17]
  #[512]
  var12972=tf.reshape(var12971, [512])
  #[512,12]
  var12973=tf.gather(params=var12410, indices=var12972, batch_dims=0, axis=0)
  #[512,12]
  var12974=tf.multiply(var12409, var12973)
  #[512,172]
  var12975=tf.concat([var12970,var12974], axis=1)
  #[512,172]
  var12976=tf.reshape(var12975, [512,172])
  #[512,160]
  var12977=tf.matmul(var12976, var12418)
  #[512,160]
  var12978=tf.reshape(var12977, [512,160])
  #[512,160]
  var12979=tf.add(var12978, var12422)
  #[512,160]
  var12980=tf.sigmoid(var12979)
  #[512,160]
  var12981=tf.matmul(var12976, var12425)
  #[512,160]
  var12982=tf.reshape(var12981, [512,160])
  #[512,160]
  var12983=tf.add(var12982, var12429)
  #[512,160]
  var12984=tf.sigmoid(var12983)
  #[512,160]
  var12985=tf.multiply(var12984, var12961)
  #[512,160]
  var12986=tf.matmul(var12976, var12433)
  #[512,160]
  var12987=tf.reshape(var12986, [512,160])
  #[512,160]
  var12988=tf.add(var12987, var12437)
  #[512,160]
  var12989=tf.sigmoid(var12988)
  #[512,160]
  var12990=tf.matmul(var12976, var12440)
  #[512,160]
  var12991=tf.reshape(var12990, [512,160])
  #[512,160]
  var12992=tf.add(var12991, var12444)
  #[512,160]
  var12993=tf.tanh(var12992)
  #[512,160]
  var12994=tf.multiply(var12989, var12993)
  #[512,160]
  var12995=tf.add(var12985, var12994)
  #[512,160]
  var12996=tf.tanh(var12995)
  #[512,160]
  var12997=tf.multiply(var12980, var12996)
  #[512,160]
  var12998=tf.multiply(var12390, var12997)
  #[512,160]
  var12999=tf.reshape(var12998, [512,160])
  #[512,12]
  var13000=tf.matmul(var12999, var12453)
  #[512,12]
  var13001=tf.reshape(var13000, [512,12])
  #[512,12]
  var13002=tf.add(var13001, var12457)
  #[512,1,12]
  var13003=tf.reshape(var13002, [512,1,12])
  #[512,160]
  var13004=tf.multiply(var12394, var12997)
  #[512,1]
  var13005=var12411[:,17:18]
  #[512]
  var13006=tf.reshape(var13005, [512])
  #[512,12]
  var13007=tf.gather(params=var12410, indices=var13006, batch_dims=0, axis=0)
  #[512,12]
  var13008=tf.multiply(var12409, var13007)
  #[512,172]
  var13009=tf.concat([var13004,var13008], axis=1)
  #[512,172]
  var13010=tf.reshape(var13009, [512,172])
  #[512,160]
  var13011=tf.matmul(var13010, var12418)
  #[512,160]
  var13012=tf.reshape(var13011, [512,160])
  #[512,160]
  var13013=tf.add(var13012, var12422)
  #[512,160]
  var13014=tf.sigmoid(var13013)
  #[512,160]
  var13015=tf.matmul(var13010, var12425)
  #[512,160]
  var13016=tf.reshape(var13015, [512,160])
  #[512,160]
  var13017=tf.add(var13016, var12429)
  #[512,160]
  var13018=tf.sigmoid(var13017)
  #[512,160]
  var13019=tf.multiply(var13018, var12995)
  #[512,160]
  var13020=tf.matmul(var13010, var12433)
  #[512,160]
  var13021=tf.reshape(var13020, [512,160])
  #[512,160]
  var13022=tf.add(var13021, var12437)
  #[512,160]
  var13023=tf.sigmoid(var13022)
  #[512,160]
  var13024=tf.matmul(var13010, var12440)
  #[512,160]
  var13025=tf.reshape(var13024, [512,160])
  #[512,160]
  var13026=tf.add(var13025, var12444)
  #[512,160]
  var13027=tf.tanh(var13026)
  #[512,160]
  var13028=tf.multiply(var13023, var13027)
  #[512,160]
  var13029=tf.add(var13019, var13028)
  #[512,160]
  var13030=tf.tanh(var13029)
  #[512,160]
  var13031=tf.multiply(var13014, var13030)
  #[512,160]
  var13032=tf.multiply(var12390, var13031)
  #[512,160]
  var13033=tf.reshape(var13032, [512,160])
  #[512,12]
  var13034=tf.matmul(var13033, var12453)
  #[512,12]
  var13035=tf.reshape(var13034, [512,12])
  #[512,12]
  var13036=tf.add(var13035, var12457)
  #[512,1,12]
  var13037=tf.reshape(var13036, [512,1,12])
  #[512,160]
  var13038=tf.multiply(var12394, var13031)
  #[512,1]
  var13039=var12411[:,18:19]
  #[512]
  var13040=tf.reshape(var13039, [512])
  #[512,12]
  var13041=tf.gather(params=var12410, indices=var13040, batch_dims=0, axis=0)
  #[512,12]
  var13042=tf.multiply(var12409, var13041)
  #[512,172]
  var13043=tf.concat([var13038,var13042], axis=1)
  #[512,172]
  var13044=tf.reshape(var13043, [512,172])
  #[512,160]
  var13045=tf.matmul(var13044, var12418)
  #[512,160]
  var13046=tf.reshape(var13045, [512,160])
  #[512,160]
  var13047=tf.add(var13046, var12422)
  #[512,160]
  var13048=tf.sigmoid(var13047)
  #[512,160]
  var13049=tf.matmul(var13044, var12425)
  #[512,160]
  var13050=tf.reshape(var13049, [512,160])
  #[512,160]
  var13051=tf.add(var13050, var12429)
  #[512,160]
  var13052=tf.sigmoid(var13051)
  #[512,160]
  var13053=tf.multiply(var13052, var13029)
  #[512,160]
  var13054=tf.matmul(var13044, var12433)
  #[512,160]
  var13055=tf.reshape(var13054, [512,160])
  #[512,160]
  var13056=tf.add(var13055, var12437)
  #[512,160]
  var13057=tf.sigmoid(var13056)
  #[512,160]
  var13058=tf.matmul(var13044, var12440)
  #[512,160]
  var13059=tf.reshape(var13058, [512,160])
  #[512,160]
  var13060=tf.add(var13059, var12444)
  #[512,160]
  var13061=tf.tanh(var13060)
  #[512,160]
  var13062=tf.multiply(var13057, var13061)
  #[512,160]
  var13063=tf.add(var13053, var13062)
  #[512,160]
  var13064=tf.tanh(var13063)
  #[512,160]
  var13065=tf.multiply(var13048, var13064)
  #[512,160]
  var13066=tf.multiply(var12390, var13065)
  #[512,160]
  var13067=tf.reshape(var13066, [512,160])
  #[512,12]
  var13068=tf.matmul(var13067, var12453)
  #[512,12]
  var13069=tf.reshape(var13068, [512,12])
  #[512,12]
  var13070=tf.add(var13069, var12457)
  #[512,1,12]
  var13071=tf.reshape(var13070, [512,1,12])
  #[512,160]
  var13072=tf.multiply(var12394, var13065)
  #[512,1]
  var13073=var12411[:,19:20]
  #[512]
  var13074=tf.reshape(var13073, [512])
  #[512,12]
  var13075=tf.gather(params=var12410, indices=var13074, batch_dims=0, axis=0)
  #[512,12]
  var13076=tf.multiply(var12409, var13075)
  #[512,172]
  var13077=tf.concat([var13072,var13076], axis=1)
  #[512,172]
  var13078=tf.reshape(var13077, [512,172])
  #[512,160]
  var13079=tf.matmul(var13078, var12418)
  #[512,160]
  var13080=tf.reshape(var13079, [512,160])
  #[512,160]
  var13081=tf.add(var13080, var12422)
  #[512,160]
  var13082=tf.sigmoid(var13081)
  #[512,160]
  var13083=tf.matmul(var13078, var12425)
  #[512,160]
  var13084=tf.reshape(var13083, [512,160])
  #[512,160]
  var13085=tf.add(var13084, var12429)
  #[512,160]
  var13086=tf.sigmoid(var13085)
  #[512,160]
  var13087=tf.multiply(var13086, var13063)
  #[512,160]
  var13088=tf.matmul(var13078, var12433)
  #[512,160]
  var13089=tf.reshape(var13088, [512,160])
  #[512,160]
  var13090=tf.add(var13089, var12437)
  #[512,160]
  var13091=tf.sigmoid(var13090)
  #[512,160]
  var13092=tf.matmul(var13078, var12440)
  #[512,160]
  var13093=tf.reshape(var13092, [512,160])
  #[512,160]
  var13094=tf.add(var13093, var12444)
  #[512,160]
  var13095=tf.tanh(var13094)
  #[512,160]
  var13096=tf.multiply(var13091, var13095)
  #[512,160]
  var13097=tf.add(var13087, var13096)
  #[512,160]
  var13098=tf.tanh(var13097)
  #[512,160]
  var13099=tf.multiply(var13082, var13098)
  #[512,160]
  var13100=tf.multiply(var12390, var13099)
  #[512,160]
  var13101=tf.reshape(var13100, [512,160])
  #[512,12]
  var13102=tf.matmul(var13101, var12453)
  #[512,12]
  var13103=tf.reshape(var13102, [512,12])
  #[512,12]
  var13104=tf.add(var13103, var12457)
  #[512,1,12]
  var13105=tf.reshape(var13104, [512,1,12])
  #[512,160]
  var13106=tf.multiply(var12394, var13099)
  #[512,1]
  var13107=var12411[:,20:21]
  #[512]
  var13108=tf.reshape(var13107, [512])
  #[512,12]
  var13109=tf.gather(params=var12410, indices=var13108, batch_dims=0, axis=0)
  #[512,12]
  var13110=tf.multiply(var12409, var13109)
  #[512,172]
  var13111=tf.concat([var13106,var13110], axis=1)
  #[512,172]
  var13112=tf.reshape(var13111, [512,172])
  #[512,160]
  var13113=tf.matmul(var13112, var12418)
  #[512,160]
  var13114=tf.reshape(var13113, [512,160])
  #[512,160]
  var13115=tf.add(var13114, var12422)
  #[512,160]
  var13116=tf.sigmoid(var13115)
  #[512,160]
  var13117=tf.matmul(var13112, var12425)
  #[512,160]
  var13118=tf.reshape(var13117, [512,160])
  #[512,160]
  var13119=tf.add(var13118, var12429)
  #[512,160]
  var13120=tf.sigmoid(var13119)
  #[512,160]
  var13121=tf.multiply(var13120, var13097)
  #[512,160]
  var13122=tf.matmul(var13112, var12433)
  #[512,160]
  var13123=tf.reshape(var13122, [512,160])
  #[512,160]
  var13124=tf.add(var13123, var12437)
  #[512,160]
  var13125=tf.sigmoid(var13124)
  #[512,160]
  var13126=tf.matmul(var13112, var12440)
  #[512,160]
  var13127=tf.reshape(var13126, [512,160])
  #[512,160]
  var13128=tf.add(var13127, var12444)
  #[512,160]
  var13129=tf.tanh(var13128)
  #[512,160]
  var13130=tf.multiply(var13125, var13129)
  #[512,160]
  var13131=tf.add(var13121, var13130)
  #[512,160]
  var13132=tf.tanh(var13131)
  #[512,160]
  var13133=tf.multiply(var13116, var13132)
  #[512,160]
  var13134=tf.multiply(var12390, var13133)
  #[512,160]
  var13135=tf.reshape(var13134, [512,160])
  #[512,12]
  var13136=tf.matmul(var13135, var12453)
  #[512,12]
  var13137=tf.reshape(var13136, [512,12])
  #[512,12]
  var13138=tf.add(var13137, var12457)
  #[512,1,12]
  var13139=tf.reshape(var13138, [512,1,12])
  #[512,21,12]
  var13140=tf.concat([var12459
                     ,var12493
                     ,var12527
                     ,var12561
                     ,var12595
                     ,var12629
                     ,var12663
                     ,var12697
                     ,var12731
                     ,var12765
                     ,var12799
                     ,var12833
                     ,var12867
                     ,var12901
                     ,var12935
                     ,var12969
                     ,var13003
                     ,var13037
                     ,var13071
                     ,var13105
                     ,var13139],
                     axis=1)
  #[512,21]
  var13141=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=var12377, logits=var13140)
  #[512,21]
  var13142=weights
  #[512,21]
  var13143=tf.multiply(var13141, var13142)
  #[512,21]
  var13144=tf.reshape(var13143, [512,21])
  #[512]
  var13145=tf.reduce_sum(var13144, axis=1)
  #[512,21]
  var13146=tf.reshape(var13142, [512,21])
  #[512]
  var13147=tf.reduce_sum(var13146, axis=1)
  #[512]
  var13148=tf.divide(var13145, var13147)
  #[512]
  var13149=tf.cast(var13148, tf.float32)
  #[512]
  var13150=tf.reshape(var13149, [512])
  #[]
  var13151=tf.reduce_mean(var13150, axis=0)
  #[1]
  var13152=tf.broadcast_to(tf.reshape(var12395, [1]), [1])
  #[]
  var13153=tf.reshape(var13152, [])
  #[]
  var13154=tf.add(var13151, var13153)
  #[512,21]
  var13155=tf.argmax(var13140, axis=2, output_type=tf.int32)
  #[512,21]
  var13156=tf.equal(var13155, var12377)
  #[512,21]
  var13157=tf.cast(var13156, tf.float32)
  #[512,21]
  var13158=tf.multiply(var13157, var13142)
  #[512,21]
  var13159=tf.reshape(var13158, [512,21])
  #[512]
  var13160=tf.reduce_sum(var13159, axis=1)
  #[512]
  var13161=tf.divide(var13160, var13147)
  #[512]
  var13162=tf.cast(var13161, tf.float32)
  #[512]
  var13163=tf.reshape(var13162, [512])
  #[]
  var13164=tf.reduce_mean(var13163, axis=0)
  #[10752,12]
  var13165=tf.reshape(var13140, [10752,12])
  #[10752,12]
  var13166=tf.nn.softmax(var13165, axis=1)
  #[512,21,12]
  var13167=tf.reshape(var13166, [512,21,12])
  return {"loss":var13154,"accuracy":var13164,"y_":var13167}
runModel = {"function":runModel_fn
           ,"batched":True
           ,"placeholders":{"x":{"shape":[512,21],"dtype":tf.int32}
                           ,"y":{"shape":[512,21],"dtype":tf.int32}
                           ,"weights":{"shape":[512,21],"dtype":tf.float32}}}