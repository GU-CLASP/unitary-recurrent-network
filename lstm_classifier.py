
import tensorflow as tf
def mkModel():
  
  #[50050,12]
  var12345=tf.random.uniform(
             [50050,12], minval=-5.0e-2, maxval=5.0e-2, dtype=tf.float32) # 0
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
  #[160,2]
  var12373=tf.random.uniform(
             [160,2], minval=-0.19245009, maxval=0.19245009, dtype=tf.float32) # 13
  var12374=tf.Variable(name="dense_w", trainable=True, initial_value=var12373)
  #[2]
  var12375=tf.random.truncated_normal([2], stddev=0.1, dtype=tf.float32) # 14
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
                yIndex,
                y):
  
  #[512]
  var12377=y
  #[512,2]
  var12378=tf.one_hot(var12377, axis=1, dtype=tf.float32, depth=2)
  #[]
  var12379=training_placeholder
  #[512,160]
  var12380=tf.random.uniform([512,160], minval=0.9, maxval=1.9, dtype=tf.float32) # 11
  #[512,160]
  var12381=tf.floor(var12380)
  #[]
  var12382=tf.constant(0.9, shape=[], dtype=tf.float32)
  #[160]
  var12383=tf.broadcast_to(tf.reshape(var12382, [1]), [160])
  #[160]
  var12384=tf.reshape(var12383, [160])
  #[512,160]
  var12385=tf.broadcast_to(tf.reshape(var12384, [1,160]), [512,160])
  #[512,160]
  var12386=tf.divide(var12381, var12385)
  #[]
  var12387=tf.constant(1.0, shape=[], dtype=tf.float32)
  #[160]
  var12388=tf.broadcast_to(tf.reshape(var12387, [1]), [160])
  #[160]
  var12389=tf.reshape(var12388, [160])
  #[512,160]
  var12390=tf.broadcast_to(tf.reshape(var12389, [1,160]), [512,160])
  #[512,160]
  var12391=tf.cond(var12379, true_fn=lambda: var12386, false_fn=lambda: var12390)
  #[512,160]
  var12392=tf.random.uniform([512,160], minval=0.9, maxval=1.9, dtype=tf.float32) # 10
  #[512,160]
  var12393=tf.floor(var12392)
  #[512,160]
  var12394=tf.divide(var12393, var12385)
  #[512,160]
  var12395=tf.cond(var12379, true_fn=lambda: var12394, false_fn=lambda: var12390)
  #[]
  var12396=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[160]
  var12397=tf.broadcast_to(tf.reshape(var12396, [1]), [160])
  #[160]
  var12398=tf.reshape(var12397, [160])
  #[512,160]
  var12399=tf.broadcast_to(tf.reshape(var12398, [1,160]), [512,160])
  #[512,160]
  var12400=tf.multiply(var12395, var12399)
  #[512,12]
  var12401=tf.random.uniform([512,12], minval=0.9, maxval=1.9, dtype=tf.float32) # 9
  #[512,12]
  var12402=tf.floor(var12401)
  #[12]
  var12403=tf.broadcast_to(tf.reshape(var12382, [1]), [12])
  #[12]
  var12404=tf.reshape(var12403, [12])
  #[512,12]
  var12405=tf.broadcast_to(tf.reshape(var12404, [1,12]), [512,12])
  #[512,12]
  var12406=tf.divide(var12402, var12405)
  #[12]
  var12407=tf.broadcast_to(tf.reshape(var12387, [1]), [12])
  #[12]
  var12408=tf.reshape(var12407, [12])
  #[512,12]
  var12409=tf.broadcast_to(tf.reshape(var12408, [1,12]), [512,12])
  #[512,12]
  var12410=tf.cond(var12379, true_fn=lambda: var12406, false_fn=lambda: var12409)
  #[512,12]
  var12411=tf.random.uniform([512,12], minval=0.9, maxval=1.9, dtype=tf.float32) # 12
  #[512,12]
  var12412=tf.floor(var12411)
  #[512,12]
  var12413=tf.divide(var12412, var12405)
  #[512,12]
  var12414=tf.cond(var12379, true_fn=lambda: var12413, false_fn=lambda: var12409)
  #[50050,12]
  var12415=embs
  #[512,50]
  var12416=x
  #[512,1]
  var12417=var12416[:,0:1]
  #[512]
  var12418=tf.reshape(var12417, [512])
  #[512,12]
  var12419=tf.gather(params=var12415, indices=var12418, batch_dims=0, axis=0)
  #[512,12]
  var12420=tf.multiply(var12414, var12419)
  #[512,12]
  var12421=tf.multiply(var12410, var12420)
  #[512,172]
  var12422=tf.concat([var12400,var12421], axis=1)
  #[512,172]
  var12423=tf.reshape(var12422, [512,172])
  #[172,160]
  var12424=w1_o_w
  #[512,160]
  var12425=tf.matmul(var12423, var12424)
  #[512,160]
  var12426=tf.reshape(var12425, [512,160])
  #[160]
  var12427=w1_o_bias
  #[512,160]
  var12428=tf.broadcast_to(tf.reshape(var12427, [1,160]), [512,160])
  #[512,160]
  var12429=tf.add(var12426, var12428)
  #[512,160]
  var12430=tf.sigmoid(var12429)
  #[172,160]
  var12431=w1_f_w
  #[512,160]
  var12432=tf.matmul(var12423, var12431)
  #[512,160]
  var12433=tf.reshape(var12432, [512,160])
  #[160]
  var12434=w1_f_bias
  #[512,160]
  var12435=tf.broadcast_to(tf.reshape(var12434, [1,160]), [512,160])
  #[512,160]
  var12436=tf.add(var12433, var12435)
  #[512,160]
  var12437=tf.sigmoid(var12436)
  #[512,160]
  var12438=tf.multiply(var12437, var12399)
  #[172,160]
  var12439=w1_i_w
  #[512,160]
  var12440=tf.matmul(var12423, var12439)
  #[512,160]
  var12441=tf.reshape(var12440, [512,160])
  #[160]
  var12442=w1_i_bias
  #[512,160]
  var12443=tf.broadcast_to(tf.reshape(var12442, [1,160]), [512,160])
  #[512,160]
  var12444=tf.add(var12441, var12443)
  #[512,160]
  var12445=tf.sigmoid(var12444)
  #[172,160]
  var12446=w1_c_w
  #[512,160]
  var12447=tf.matmul(var12423, var12446)
  #[512,160]
  var12448=tf.reshape(var12447, [512,160])
  #[160]
  var12449=w1_c_bias
  #[512,160]
  var12450=tf.broadcast_to(tf.reshape(var12449, [1,160]), [512,160])
  #[512,160]
  var12451=tf.add(var12448, var12450)
  #[512,160]
  var12452=tf.tanh(var12451)
  #[512,160]
  var12453=tf.multiply(var12445, var12452)
  #[512,160]
  var12454=tf.add(var12438, var12453)
  #[512,160]
  var12455=tf.tanh(var12454)
  #[512,160]
  var12456=tf.multiply(var12430, var12455)
  #[512,160]
  var12457=tf.multiply(var12391, var12456)
  #[512,160]
  var12458=tf.reshape(var12457, [512,160])
  #[160,2]
  var12459=dense_w
  #[512,2]
  var12460=tf.matmul(var12458, var12459)
  #[512,2]
  var12461=tf.reshape(var12460, [512,2])
  #[2]
  var12462=dense_bias
  #[512,2]
  var12463=tf.broadcast_to(tf.reshape(var12462, [1,2]), [512,2])
  #[512,2]
  var12464=tf.add(var12461, var12463)
  #[512,1,2]
  var12465=tf.reshape(var12464, [512,1,2])
  #[512,160]
  var12466=tf.multiply(var12395, var12456)
  #[512,1]
  var12467=var12416[:,1:2]
  #[512]
  var12468=tf.reshape(var12467, [512])
  #[512,12]
  var12469=tf.gather(params=var12415, indices=var12468, batch_dims=0, axis=0)
  #[512,12]
  var12470=tf.multiply(var12414, var12469)
  #[512,12]
  var12471=tf.multiply(var12410, var12470)
  #[512,172]
  var12472=tf.concat([var12466,var12471], axis=1)
  #[512,172]
  var12473=tf.reshape(var12472, [512,172])
  #[512,160]
  var12474=tf.matmul(var12473, var12424)
  #[512,160]
  var12475=tf.reshape(var12474, [512,160])
  #[512,160]
  var12476=tf.add(var12475, var12428)
  #[512,160]
  var12477=tf.sigmoid(var12476)
  #[512,160]
  var12478=tf.matmul(var12473, var12431)
  #[512,160]
  var12479=tf.reshape(var12478, [512,160])
  #[512,160]
  var12480=tf.add(var12479, var12435)
  #[512,160]
  var12481=tf.sigmoid(var12480)
  #[512,160]
  var12482=tf.multiply(var12481, var12454)
  #[512,160]
  var12483=tf.matmul(var12473, var12439)
  #[512,160]
  var12484=tf.reshape(var12483, [512,160])
  #[512,160]
  var12485=tf.add(var12484, var12443)
  #[512,160]
  var12486=tf.sigmoid(var12485)
  #[512,160]
  var12487=tf.matmul(var12473, var12446)
  #[512,160]
  var12488=tf.reshape(var12487, [512,160])
  #[512,160]
  var12489=tf.add(var12488, var12450)
  #[512,160]
  var12490=tf.tanh(var12489)
  #[512,160]
  var12491=tf.multiply(var12486, var12490)
  #[512,160]
  var12492=tf.add(var12482, var12491)
  #[512,160]
  var12493=tf.tanh(var12492)
  #[512,160]
  var12494=tf.multiply(var12477, var12493)
  #[512,160]
  var12495=tf.multiply(var12391, var12494)
  #[512,160]
  var12496=tf.reshape(var12495, [512,160])
  #[512,2]
  var12497=tf.matmul(var12496, var12459)
  #[512,2]
  var12498=tf.reshape(var12497, [512,2])
  #[512,2]
  var12499=tf.add(var12498, var12463)
  #[512,1,2]
  var12500=tf.reshape(var12499, [512,1,2])
  #[512,160]
  var12501=tf.multiply(var12395, var12494)
  #[512,1]
  var12502=var12416[:,2:3]
  #[512]
  var12503=tf.reshape(var12502, [512])
  #[512,12]
  var12504=tf.gather(params=var12415, indices=var12503, batch_dims=0, axis=0)
  #[512,12]
  var12505=tf.multiply(var12414, var12504)
  #[512,12]
  var12506=tf.multiply(var12410, var12505)
  #[512,172]
  var12507=tf.concat([var12501,var12506], axis=1)
  #[512,172]
  var12508=tf.reshape(var12507, [512,172])
  #[512,160]
  var12509=tf.matmul(var12508, var12424)
  #[512,160]
  var12510=tf.reshape(var12509, [512,160])
  #[512,160]
  var12511=tf.add(var12510, var12428)
  #[512,160]
  var12512=tf.sigmoid(var12511)
  #[512,160]
  var12513=tf.matmul(var12508, var12431)
  #[512,160]
  var12514=tf.reshape(var12513, [512,160])
  #[512,160]
  var12515=tf.add(var12514, var12435)
  #[512,160]
  var12516=tf.sigmoid(var12515)
  #[512,160]
  var12517=tf.multiply(var12516, var12492)
  #[512,160]
  var12518=tf.matmul(var12508, var12439)
  #[512,160]
  var12519=tf.reshape(var12518, [512,160])
  #[512,160]
  var12520=tf.add(var12519, var12443)
  #[512,160]
  var12521=tf.sigmoid(var12520)
  #[512,160]
  var12522=tf.matmul(var12508, var12446)
  #[512,160]
  var12523=tf.reshape(var12522, [512,160])
  #[512,160]
  var12524=tf.add(var12523, var12450)
  #[512,160]
  var12525=tf.tanh(var12524)
  #[512,160]
  var12526=tf.multiply(var12521, var12525)
  #[512,160]
  var12527=tf.add(var12517, var12526)
  #[512,160]
  var12528=tf.tanh(var12527)
  #[512,160]
  var12529=tf.multiply(var12512, var12528)
  #[512,160]
  var12530=tf.multiply(var12391, var12529)
  #[512,160]
  var12531=tf.reshape(var12530, [512,160])
  #[512,2]
  var12532=tf.matmul(var12531, var12459)
  #[512,2]
  var12533=tf.reshape(var12532, [512,2])
  #[512,2]
  var12534=tf.add(var12533, var12463)
  #[512,1,2]
  var12535=tf.reshape(var12534, [512,1,2])
  #[512,160]
  var12536=tf.multiply(var12395, var12529)
  #[512,1]
  var12537=var12416[:,3:4]
  #[512]
  var12538=tf.reshape(var12537, [512])
  #[512,12]
  var12539=tf.gather(params=var12415, indices=var12538, batch_dims=0, axis=0)
  #[512,12]
  var12540=tf.multiply(var12414, var12539)
  #[512,12]
  var12541=tf.multiply(var12410, var12540)
  #[512,172]
  var12542=tf.concat([var12536,var12541], axis=1)
  #[512,172]
  var12543=tf.reshape(var12542, [512,172])
  #[512,160]
  var12544=tf.matmul(var12543, var12424)
  #[512,160]
  var12545=tf.reshape(var12544, [512,160])
  #[512,160]
  var12546=tf.add(var12545, var12428)
  #[512,160]
  var12547=tf.sigmoid(var12546)
  #[512,160]
  var12548=tf.matmul(var12543, var12431)
  #[512,160]
  var12549=tf.reshape(var12548, [512,160])
  #[512,160]
  var12550=tf.add(var12549, var12435)
  #[512,160]
  var12551=tf.sigmoid(var12550)
  #[512,160]
  var12552=tf.multiply(var12551, var12527)
  #[512,160]
  var12553=tf.matmul(var12543, var12439)
  #[512,160]
  var12554=tf.reshape(var12553, [512,160])
  #[512,160]
  var12555=tf.add(var12554, var12443)
  #[512,160]
  var12556=tf.sigmoid(var12555)
  #[512,160]
  var12557=tf.matmul(var12543, var12446)
  #[512,160]
  var12558=tf.reshape(var12557, [512,160])
  #[512,160]
  var12559=tf.add(var12558, var12450)
  #[512,160]
  var12560=tf.tanh(var12559)
  #[512,160]
  var12561=tf.multiply(var12556, var12560)
  #[512,160]
  var12562=tf.add(var12552, var12561)
  #[512,160]
  var12563=tf.tanh(var12562)
  #[512,160]
  var12564=tf.multiply(var12547, var12563)
  #[512,160]
  var12565=tf.multiply(var12391, var12564)
  #[512,160]
  var12566=tf.reshape(var12565, [512,160])
  #[512,2]
  var12567=tf.matmul(var12566, var12459)
  #[512,2]
  var12568=tf.reshape(var12567, [512,2])
  #[512,2]
  var12569=tf.add(var12568, var12463)
  #[512,1,2]
  var12570=tf.reshape(var12569, [512,1,2])
  #[512,160]
  var12571=tf.multiply(var12395, var12564)
  #[512,1]
  var12572=var12416[:,4:5]
  #[512]
  var12573=tf.reshape(var12572, [512])
  #[512,12]
  var12574=tf.gather(params=var12415, indices=var12573, batch_dims=0, axis=0)
  #[512,12]
  var12575=tf.multiply(var12414, var12574)
  #[512,12]
  var12576=tf.multiply(var12410, var12575)
  #[512,172]
  var12577=tf.concat([var12571,var12576], axis=1)
  #[512,172]
  var12578=tf.reshape(var12577, [512,172])
  #[512,160]
  var12579=tf.matmul(var12578, var12424)
  #[512,160]
  var12580=tf.reshape(var12579, [512,160])
  #[512,160]
  var12581=tf.add(var12580, var12428)
  #[512,160]
  var12582=tf.sigmoid(var12581)
  #[512,160]
  var12583=tf.matmul(var12578, var12431)
  #[512,160]
  var12584=tf.reshape(var12583, [512,160])
  #[512,160]
  var12585=tf.add(var12584, var12435)
  #[512,160]
  var12586=tf.sigmoid(var12585)
  #[512,160]
  var12587=tf.multiply(var12586, var12562)
  #[512,160]
  var12588=tf.matmul(var12578, var12439)
  #[512,160]
  var12589=tf.reshape(var12588, [512,160])
  #[512,160]
  var12590=tf.add(var12589, var12443)
  #[512,160]
  var12591=tf.sigmoid(var12590)
  #[512,160]
  var12592=tf.matmul(var12578, var12446)
  #[512,160]
  var12593=tf.reshape(var12592, [512,160])
  #[512,160]
  var12594=tf.add(var12593, var12450)
  #[512,160]
  var12595=tf.tanh(var12594)
  #[512,160]
  var12596=tf.multiply(var12591, var12595)
  #[512,160]
  var12597=tf.add(var12587, var12596)
  #[512,160]
  var12598=tf.tanh(var12597)
  #[512,160]
  var12599=tf.multiply(var12582, var12598)
  #[512,160]
  var12600=tf.multiply(var12391, var12599)
  #[512,160]
  var12601=tf.reshape(var12600, [512,160])
  #[512,2]
  var12602=tf.matmul(var12601, var12459)
  #[512,2]
  var12603=tf.reshape(var12602, [512,2])
  #[512,2]
  var12604=tf.add(var12603, var12463)
  #[512,1,2]
  var12605=tf.reshape(var12604, [512,1,2])
  #[512,160]
  var12606=tf.multiply(var12395, var12599)
  #[512,1]
  var12607=var12416[:,5:6]
  #[512]
  var12608=tf.reshape(var12607, [512])
  #[512,12]
  var12609=tf.gather(params=var12415, indices=var12608, batch_dims=0, axis=0)
  #[512,12]
  var12610=tf.multiply(var12414, var12609)
  #[512,12]
  var12611=tf.multiply(var12410, var12610)
  #[512,172]
  var12612=tf.concat([var12606,var12611], axis=1)
  #[512,172]
  var12613=tf.reshape(var12612, [512,172])
  #[512,160]
  var12614=tf.matmul(var12613, var12424)
  #[512,160]
  var12615=tf.reshape(var12614, [512,160])
  #[512,160]
  var12616=tf.add(var12615, var12428)
  #[512,160]
  var12617=tf.sigmoid(var12616)
  #[512,160]
  var12618=tf.matmul(var12613, var12431)
  #[512,160]
  var12619=tf.reshape(var12618, [512,160])
  #[512,160]
  var12620=tf.add(var12619, var12435)
  #[512,160]
  var12621=tf.sigmoid(var12620)
  #[512,160]
  var12622=tf.multiply(var12621, var12597)
  #[512,160]
  var12623=tf.matmul(var12613, var12439)
  #[512,160]
  var12624=tf.reshape(var12623, [512,160])
  #[512,160]
  var12625=tf.add(var12624, var12443)
  #[512,160]
  var12626=tf.sigmoid(var12625)
  #[512,160]
  var12627=tf.matmul(var12613, var12446)
  #[512,160]
  var12628=tf.reshape(var12627, [512,160])
  #[512,160]
  var12629=tf.add(var12628, var12450)
  #[512,160]
  var12630=tf.tanh(var12629)
  #[512,160]
  var12631=tf.multiply(var12626, var12630)
  #[512,160]
  var12632=tf.add(var12622, var12631)
  #[512,160]
  var12633=tf.tanh(var12632)
  #[512,160]
  var12634=tf.multiply(var12617, var12633)
  #[512,160]
  var12635=tf.multiply(var12391, var12634)
  #[512,160]
  var12636=tf.reshape(var12635, [512,160])
  #[512,2]
  var12637=tf.matmul(var12636, var12459)
  #[512,2]
  var12638=tf.reshape(var12637, [512,2])
  #[512,2]
  var12639=tf.add(var12638, var12463)
  #[512,1,2]
  var12640=tf.reshape(var12639, [512,1,2])
  #[512,160]
  var12641=tf.multiply(var12395, var12634)
  #[512,1]
  var12642=var12416[:,6:7]
  #[512]
  var12643=tf.reshape(var12642, [512])
  #[512,12]
  var12644=tf.gather(params=var12415, indices=var12643, batch_dims=0, axis=0)
  #[512,12]
  var12645=tf.multiply(var12414, var12644)
  #[512,12]
  var12646=tf.multiply(var12410, var12645)
  #[512,172]
  var12647=tf.concat([var12641,var12646], axis=1)
  #[512,172]
  var12648=tf.reshape(var12647, [512,172])
  #[512,160]
  var12649=tf.matmul(var12648, var12424)
  #[512,160]
  var12650=tf.reshape(var12649, [512,160])
  #[512,160]
  var12651=tf.add(var12650, var12428)
  #[512,160]
  var12652=tf.sigmoid(var12651)
  #[512,160]
  var12653=tf.matmul(var12648, var12431)
  #[512,160]
  var12654=tf.reshape(var12653, [512,160])
  #[512,160]
  var12655=tf.add(var12654, var12435)
  #[512,160]
  var12656=tf.sigmoid(var12655)
  #[512,160]
  var12657=tf.multiply(var12656, var12632)
  #[512,160]
  var12658=tf.matmul(var12648, var12439)
  #[512,160]
  var12659=tf.reshape(var12658, [512,160])
  #[512,160]
  var12660=tf.add(var12659, var12443)
  #[512,160]
  var12661=tf.sigmoid(var12660)
  #[512,160]
  var12662=tf.matmul(var12648, var12446)
  #[512,160]
  var12663=tf.reshape(var12662, [512,160])
  #[512,160]
  var12664=tf.add(var12663, var12450)
  #[512,160]
  var12665=tf.tanh(var12664)
  #[512,160]
  var12666=tf.multiply(var12661, var12665)
  #[512,160]
  var12667=tf.add(var12657, var12666)
  #[512,160]
  var12668=tf.tanh(var12667)
  #[512,160]
  var12669=tf.multiply(var12652, var12668)
  #[512,160]
  var12670=tf.multiply(var12391, var12669)
  #[512,160]
  var12671=tf.reshape(var12670, [512,160])
  #[512,2]
  var12672=tf.matmul(var12671, var12459)
  #[512,2]
  var12673=tf.reshape(var12672, [512,2])
  #[512,2]
  var12674=tf.add(var12673, var12463)
  #[512,1,2]
  var12675=tf.reshape(var12674, [512,1,2])
  #[512,160]
  var12676=tf.multiply(var12395, var12669)
  #[512,1]
  var12677=var12416[:,7:8]
  #[512]
  var12678=tf.reshape(var12677, [512])
  #[512,12]
  var12679=tf.gather(params=var12415, indices=var12678, batch_dims=0, axis=0)
  #[512,12]
  var12680=tf.multiply(var12414, var12679)
  #[512,12]
  var12681=tf.multiply(var12410, var12680)
  #[512,172]
  var12682=tf.concat([var12676,var12681], axis=1)
  #[512,172]
  var12683=tf.reshape(var12682, [512,172])
  #[512,160]
  var12684=tf.matmul(var12683, var12424)
  #[512,160]
  var12685=tf.reshape(var12684, [512,160])
  #[512,160]
  var12686=tf.add(var12685, var12428)
  #[512,160]
  var12687=tf.sigmoid(var12686)
  #[512,160]
  var12688=tf.matmul(var12683, var12431)
  #[512,160]
  var12689=tf.reshape(var12688, [512,160])
  #[512,160]
  var12690=tf.add(var12689, var12435)
  #[512,160]
  var12691=tf.sigmoid(var12690)
  #[512,160]
  var12692=tf.multiply(var12691, var12667)
  #[512,160]
  var12693=tf.matmul(var12683, var12439)
  #[512,160]
  var12694=tf.reshape(var12693, [512,160])
  #[512,160]
  var12695=tf.add(var12694, var12443)
  #[512,160]
  var12696=tf.sigmoid(var12695)
  #[512,160]
  var12697=tf.matmul(var12683, var12446)
  #[512,160]
  var12698=tf.reshape(var12697, [512,160])
  #[512,160]
  var12699=tf.add(var12698, var12450)
  #[512,160]
  var12700=tf.tanh(var12699)
  #[512,160]
  var12701=tf.multiply(var12696, var12700)
  #[512,160]
  var12702=tf.add(var12692, var12701)
  #[512,160]
  var12703=tf.tanh(var12702)
  #[512,160]
  var12704=tf.multiply(var12687, var12703)
  #[512,160]
  var12705=tf.multiply(var12391, var12704)
  #[512,160]
  var12706=tf.reshape(var12705, [512,160])
  #[512,2]
  var12707=tf.matmul(var12706, var12459)
  #[512,2]
  var12708=tf.reshape(var12707, [512,2])
  #[512,2]
  var12709=tf.add(var12708, var12463)
  #[512,1,2]
  var12710=tf.reshape(var12709, [512,1,2])
  #[512,160]
  var12711=tf.multiply(var12395, var12704)
  #[512,1]
  var12712=var12416[:,8:9]
  #[512]
  var12713=tf.reshape(var12712, [512])
  #[512,12]
  var12714=tf.gather(params=var12415, indices=var12713, batch_dims=0, axis=0)
  #[512,12]
  var12715=tf.multiply(var12414, var12714)
  #[512,12]
  var12716=tf.multiply(var12410, var12715)
  #[512,172]
  var12717=tf.concat([var12711,var12716], axis=1)
  #[512,172]
  var12718=tf.reshape(var12717, [512,172])
  #[512,160]
  var12719=tf.matmul(var12718, var12424)
  #[512,160]
  var12720=tf.reshape(var12719, [512,160])
  #[512,160]
  var12721=tf.add(var12720, var12428)
  #[512,160]
  var12722=tf.sigmoid(var12721)
  #[512,160]
  var12723=tf.matmul(var12718, var12431)
  #[512,160]
  var12724=tf.reshape(var12723, [512,160])
  #[512,160]
  var12725=tf.add(var12724, var12435)
  #[512,160]
  var12726=tf.sigmoid(var12725)
  #[512,160]
  var12727=tf.multiply(var12726, var12702)
  #[512,160]
  var12728=tf.matmul(var12718, var12439)
  #[512,160]
  var12729=tf.reshape(var12728, [512,160])
  #[512,160]
  var12730=tf.add(var12729, var12443)
  #[512,160]
  var12731=tf.sigmoid(var12730)
  #[512,160]
  var12732=tf.matmul(var12718, var12446)
  #[512,160]
  var12733=tf.reshape(var12732, [512,160])
  #[512,160]
  var12734=tf.add(var12733, var12450)
  #[512,160]
  var12735=tf.tanh(var12734)
  #[512,160]
  var12736=tf.multiply(var12731, var12735)
  #[512,160]
  var12737=tf.add(var12727, var12736)
  #[512,160]
  var12738=tf.tanh(var12737)
  #[512,160]
  var12739=tf.multiply(var12722, var12738)
  #[512,160]
  var12740=tf.multiply(var12391, var12739)
  #[512,160]
  var12741=tf.reshape(var12740, [512,160])
  #[512,2]
  var12742=tf.matmul(var12741, var12459)
  #[512,2]
  var12743=tf.reshape(var12742, [512,2])
  #[512,2]
  var12744=tf.add(var12743, var12463)
  #[512,1,2]
  var12745=tf.reshape(var12744, [512,1,2])
  #[512,160]
  var12746=tf.multiply(var12395, var12739)
  #[512,1]
  var12747=var12416[:,9:10]
  #[512]
  var12748=tf.reshape(var12747, [512])
  #[512,12]
  var12749=tf.gather(params=var12415, indices=var12748, batch_dims=0, axis=0)
  #[512,12]
  var12750=tf.multiply(var12414, var12749)
  #[512,12]
  var12751=tf.multiply(var12410, var12750)
  #[512,172]
  var12752=tf.concat([var12746,var12751], axis=1)
  #[512,172]
  var12753=tf.reshape(var12752, [512,172])
  #[512,160]
  var12754=tf.matmul(var12753, var12424)
  #[512,160]
  var12755=tf.reshape(var12754, [512,160])
  #[512,160]
  var12756=tf.add(var12755, var12428)
  #[512,160]
  var12757=tf.sigmoid(var12756)
  #[512,160]
  var12758=tf.matmul(var12753, var12431)
  #[512,160]
  var12759=tf.reshape(var12758, [512,160])
  #[512,160]
  var12760=tf.add(var12759, var12435)
  #[512,160]
  var12761=tf.sigmoid(var12760)
  #[512,160]
  var12762=tf.multiply(var12761, var12737)
  #[512,160]
  var12763=tf.matmul(var12753, var12439)
  #[512,160]
  var12764=tf.reshape(var12763, [512,160])
  #[512,160]
  var12765=tf.add(var12764, var12443)
  #[512,160]
  var12766=tf.sigmoid(var12765)
  #[512,160]
  var12767=tf.matmul(var12753, var12446)
  #[512,160]
  var12768=tf.reshape(var12767, [512,160])
  #[512,160]
  var12769=tf.add(var12768, var12450)
  #[512,160]
  var12770=tf.tanh(var12769)
  #[512,160]
  var12771=tf.multiply(var12766, var12770)
  #[512,160]
  var12772=tf.add(var12762, var12771)
  #[512,160]
  var12773=tf.tanh(var12772)
  #[512,160]
  var12774=tf.multiply(var12757, var12773)
  #[512,160]
  var12775=tf.multiply(var12391, var12774)
  #[512,160]
  var12776=tf.reshape(var12775, [512,160])
  #[512,2]
  var12777=tf.matmul(var12776, var12459)
  #[512,2]
  var12778=tf.reshape(var12777, [512,2])
  #[512,2]
  var12779=tf.add(var12778, var12463)
  #[512,1,2]
  var12780=tf.reshape(var12779, [512,1,2])
  #[512,160]
  var12781=tf.multiply(var12395, var12774)
  #[512,1]
  var12782=var12416[:,10:11]
  #[512]
  var12783=tf.reshape(var12782, [512])
  #[512,12]
  var12784=tf.gather(params=var12415, indices=var12783, batch_dims=0, axis=0)
  #[512,12]
  var12785=tf.multiply(var12414, var12784)
  #[512,12]
  var12786=tf.multiply(var12410, var12785)
  #[512,172]
  var12787=tf.concat([var12781,var12786], axis=1)
  #[512,172]
  var12788=tf.reshape(var12787, [512,172])
  #[512,160]
  var12789=tf.matmul(var12788, var12424)
  #[512,160]
  var12790=tf.reshape(var12789, [512,160])
  #[512,160]
  var12791=tf.add(var12790, var12428)
  #[512,160]
  var12792=tf.sigmoid(var12791)
  #[512,160]
  var12793=tf.matmul(var12788, var12431)
  #[512,160]
  var12794=tf.reshape(var12793, [512,160])
  #[512,160]
  var12795=tf.add(var12794, var12435)
  #[512,160]
  var12796=tf.sigmoid(var12795)
  #[512,160]
  var12797=tf.multiply(var12796, var12772)
  #[512,160]
  var12798=tf.matmul(var12788, var12439)
  #[512,160]
  var12799=tf.reshape(var12798, [512,160])
  #[512,160]
  var12800=tf.add(var12799, var12443)
  #[512,160]
  var12801=tf.sigmoid(var12800)
  #[512,160]
  var12802=tf.matmul(var12788, var12446)
  #[512,160]
  var12803=tf.reshape(var12802, [512,160])
  #[512,160]
  var12804=tf.add(var12803, var12450)
  #[512,160]
  var12805=tf.tanh(var12804)
  #[512,160]
  var12806=tf.multiply(var12801, var12805)
  #[512,160]
  var12807=tf.add(var12797, var12806)
  #[512,160]
  var12808=tf.tanh(var12807)
  #[512,160]
  var12809=tf.multiply(var12792, var12808)
  #[512,160]
  var12810=tf.multiply(var12391, var12809)
  #[512,160]
  var12811=tf.reshape(var12810, [512,160])
  #[512,2]
  var12812=tf.matmul(var12811, var12459)
  #[512,2]
  var12813=tf.reshape(var12812, [512,2])
  #[512,2]
  var12814=tf.add(var12813, var12463)
  #[512,1,2]
  var12815=tf.reshape(var12814, [512,1,2])
  #[512,160]
  var12816=tf.multiply(var12395, var12809)
  #[512,1]
  var12817=var12416[:,11:12]
  #[512]
  var12818=tf.reshape(var12817, [512])
  #[512,12]
  var12819=tf.gather(params=var12415, indices=var12818, batch_dims=0, axis=0)
  #[512,12]
  var12820=tf.multiply(var12414, var12819)
  #[512,12]
  var12821=tf.multiply(var12410, var12820)
  #[512,172]
  var12822=tf.concat([var12816,var12821], axis=1)
  #[512,172]
  var12823=tf.reshape(var12822, [512,172])
  #[512,160]
  var12824=tf.matmul(var12823, var12424)
  #[512,160]
  var12825=tf.reshape(var12824, [512,160])
  #[512,160]
  var12826=tf.add(var12825, var12428)
  #[512,160]
  var12827=tf.sigmoid(var12826)
  #[512,160]
  var12828=tf.matmul(var12823, var12431)
  #[512,160]
  var12829=tf.reshape(var12828, [512,160])
  #[512,160]
  var12830=tf.add(var12829, var12435)
  #[512,160]
  var12831=tf.sigmoid(var12830)
  #[512,160]
  var12832=tf.multiply(var12831, var12807)
  #[512,160]
  var12833=tf.matmul(var12823, var12439)
  #[512,160]
  var12834=tf.reshape(var12833, [512,160])
  #[512,160]
  var12835=tf.add(var12834, var12443)
  #[512,160]
  var12836=tf.sigmoid(var12835)
  #[512,160]
  var12837=tf.matmul(var12823, var12446)
  #[512,160]
  var12838=tf.reshape(var12837, [512,160])
  #[512,160]
  var12839=tf.add(var12838, var12450)
  #[512,160]
  var12840=tf.tanh(var12839)
  #[512,160]
  var12841=tf.multiply(var12836, var12840)
  #[512,160]
  var12842=tf.add(var12832, var12841)
  #[512,160]
  var12843=tf.tanh(var12842)
  #[512,160]
  var12844=tf.multiply(var12827, var12843)
  #[512,160]
  var12845=tf.multiply(var12391, var12844)
  #[512,160]
  var12846=tf.reshape(var12845, [512,160])
  #[512,2]
  var12847=tf.matmul(var12846, var12459)
  #[512,2]
  var12848=tf.reshape(var12847, [512,2])
  #[512,2]
  var12849=tf.add(var12848, var12463)
  #[512,1,2]
  var12850=tf.reshape(var12849, [512,1,2])
  #[512,160]
  var12851=tf.multiply(var12395, var12844)
  #[512,1]
  var12852=var12416[:,12:13]
  #[512]
  var12853=tf.reshape(var12852, [512])
  #[512,12]
  var12854=tf.gather(params=var12415, indices=var12853, batch_dims=0, axis=0)
  #[512,12]
  var12855=tf.multiply(var12414, var12854)
  #[512,12]
  var12856=tf.multiply(var12410, var12855)
  #[512,172]
  var12857=tf.concat([var12851,var12856], axis=1)
  #[512,172]
  var12858=tf.reshape(var12857, [512,172])
  #[512,160]
  var12859=tf.matmul(var12858, var12424)
  #[512,160]
  var12860=tf.reshape(var12859, [512,160])
  #[512,160]
  var12861=tf.add(var12860, var12428)
  #[512,160]
  var12862=tf.sigmoid(var12861)
  #[512,160]
  var12863=tf.matmul(var12858, var12431)
  #[512,160]
  var12864=tf.reshape(var12863, [512,160])
  #[512,160]
  var12865=tf.add(var12864, var12435)
  #[512,160]
  var12866=tf.sigmoid(var12865)
  #[512,160]
  var12867=tf.multiply(var12866, var12842)
  #[512,160]
  var12868=tf.matmul(var12858, var12439)
  #[512,160]
  var12869=tf.reshape(var12868, [512,160])
  #[512,160]
  var12870=tf.add(var12869, var12443)
  #[512,160]
  var12871=tf.sigmoid(var12870)
  #[512,160]
  var12872=tf.matmul(var12858, var12446)
  #[512,160]
  var12873=tf.reshape(var12872, [512,160])
  #[512,160]
  var12874=tf.add(var12873, var12450)
  #[512,160]
  var12875=tf.tanh(var12874)
  #[512,160]
  var12876=tf.multiply(var12871, var12875)
  #[512,160]
  var12877=tf.add(var12867, var12876)
  #[512,160]
  var12878=tf.tanh(var12877)
  #[512,160]
  var12879=tf.multiply(var12862, var12878)
  #[512,160]
  var12880=tf.multiply(var12391, var12879)
  #[512,160]
  var12881=tf.reshape(var12880, [512,160])
  #[512,2]
  var12882=tf.matmul(var12881, var12459)
  #[512,2]
  var12883=tf.reshape(var12882, [512,2])
  #[512,2]
  var12884=tf.add(var12883, var12463)
  #[512,1,2]
  var12885=tf.reshape(var12884, [512,1,2])
  #[512,160]
  var12886=tf.multiply(var12395, var12879)
  #[512,1]
  var12887=var12416[:,13:14]
  #[512]
  var12888=tf.reshape(var12887, [512])
  #[512,12]
  var12889=tf.gather(params=var12415, indices=var12888, batch_dims=0, axis=0)
  #[512,12]
  var12890=tf.multiply(var12414, var12889)
  #[512,12]
  var12891=tf.multiply(var12410, var12890)
  #[512,172]
  var12892=tf.concat([var12886,var12891], axis=1)
  #[512,172]
  var12893=tf.reshape(var12892, [512,172])
  #[512,160]
  var12894=tf.matmul(var12893, var12424)
  #[512,160]
  var12895=tf.reshape(var12894, [512,160])
  #[512,160]
  var12896=tf.add(var12895, var12428)
  #[512,160]
  var12897=tf.sigmoid(var12896)
  #[512,160]
  var12898=tf.matmul(var12893, var12431)
  #[512,160]
  var12899=tf.reshape(var12898, [512,160])
  #[512,160]
  var12900=tf.add(var12899, var12435)
  #[512,160]
  var12901=tf.sigmoid(var12900)
  #[512,160]
  var12902=tf.multiply(var12901, var12877)
  #[512,160]
  var12903=tf.matmul(var12893, var12439)
  #[512,160]
  var12904=tf.reshape(var12903, [512,160])
  #[512,160]
  var12905=tf.add(var12904, var12443)
  #[512,160]
  var12906=tf.sigmoid(var12905)
  #[512,160]
  var12907=tf.matmul(var12893, var12446)
  #[512,160]
  var12908=tf.reshape(var12907, [512,160])
  #[512,160]
  var12909=tf.add(var12908, var12450)
  #[512,160]
  var12910=tf.tanh(var12909)
  #[512,160]
  var12911=tf.multiply(var12906, var12910)
  #[512,160]
  var12912=tf.add(var12902, var12911)
  #[512,160]
  var12913=tf.tanh(var12912)
  #[512,160]
  var12914=tf.multiply(var12897, var12913)
  #[512,160]
  var12915=tf.multiply(var12391, var12914)
  #[512,160]
  var12916=tf.reshape(var12915, [512,160])
  #[512,2]
  var12917=tf.matmul(var12916, var12459)
  #[512,2]
  var12918=tf.reshape(var12917, [512,2])
  #[512,2]
  var12919=tf.add(var12918, var12463)
  #[512,1,2]
  var12920=tf.reshape(var12919, [512,1,2])
  #[512,160]
  var12921=tf.multiply(var12395, var12914)
  #[512,1]
  var12922=var12416[:,14:15]
  #[512]
  var12923=tf.reshape(var12922, [512])
  #[512,12]
  var12924=tf.gather(params=var12415, indices=var12923, batch_dims=0, axis=0)
  #[512,12]
  var12925=tf.multiply(var12414, var12924)
  #[512,12]
  var12926=tf.multiply(var12410, var12925)
  #[512,172]
  var12927=tf.concat([var12921,var12926], axis=1)
  #[512,172]
  var12928=tf.reshape(var12927, [512,172])
  #[512,160]
  var12929=tf.matmul(var12928, var12424)
  #[512,160]
  var12930=tf.reshape(var12929, [512,160])
  #[512,160]
  var12931=tf.add(var12930, var12428)
  #[512,160]
  var12932=tf.sigmoid(var12931)
  #[512,160]
  var12933=tf.matmul(var12928, var12431)
  #[512,160]
  var12934=tf.reshape(var12933, [512,160])
  #[512,160]
  var12935=tf.add(var12934, var12435)
  #[512,160]
  var12936=tf.sigmoid(var12935)
  #[512,160]
  var12937=tf.multiply(var12936, var12912)
  #[512,160]
  var12938=tf.matmul(var12928, var12439)
  #[512,160]
  var12939=tf.reshape(var12938, [512,160])
  #[512,160]
  var12940=tf.add(var12939, var12443)
  #[512,160]
  var12941=tf.sigmoid(var12940)
  #[512,160]
  var12942=tf.matmul(var12928, var12446)
  #[512,160]
  var12943=tf.reshape(var12942, [512,160])
  #[512,160]
  var12944=tf.add(var12943, var12450)
  #[512,160]
  var12945=tf.tanh(var12944)
  #[512,160]
  var12946=tf.multiply(var12941, var12945)
  #[512,160]
  var12947=tf.add(var12937, var12946)
  #[512,160]
  var12948=tf.tanh(var12947)
  #[512,160]
  var12949=tf.multiply(var12932, var12948)
  #[512,160]
  var12950=tf.multiply(var12391, var12949)
  #[512,160]
  var12951=tf.reshape(var12950, [512,160])
  #[512,2]
  var12952=tf.matmul(var12951, var12459)
  #[512,2]
  var12953=tf.reshape(var12952, [512,2])
  #[512,2]
  var12954=tf.add(var12953, var12463)
  #[512,1,2]
  var12955=tf.reshape(var12954, [512,1,2])
  #[512,160]
  var12956=tf.multiply(var12395, var12949)
  #[512,1]
  var12957=var12416[:,15:16]
  #[512]
  var12958=tf.reshape(var12957, [512])
  #[512,12]
  var12959=tf.gather(params=var12415, indices=var12958, batch_dims=0, axis=0)
  #[512,12]
  var12960=tf.multiply(var12414, var12959)
  #[512,12]
  var12961=tf.multiply(var12410, var12960)
  #[512,172]
  var12962=tf.concat([var12956,var12961], axis=1)
  #[512,172]
  var12963=tf.reshape(var12962, [512,172])
  #[512,160]
  var12964=tf.matmul(var12963, var12424)
  #[512,160]
  var12965=tf.reshape(var12964, [512,160])
  #[512,160]
  var12966=tf.add(var12965, var12428)
  #[512,160]
  var12967=tf.sigmoid(var12966)
  #[512,160]
  var12968=tf.matmul(var12963, var12431)
  #[512,160]
  var12969=tf.reshape(var12968, [512,160])
  #[512,160]
  var12970=tf.add(var12969, var12435)
  #[512,160]
  var12971=tf.sigmoid(var12970)
  #[512,160]
  var12972=tf.multiply(var12971, var12947)
  #[512,160]
  var12973=tf.matmul(var12963, var12439)
  #[512,160]
  var12974=tf.reshape(var12973, [512,160])
  #[512,160]
  var12975=tf.add(var12974, var12443)
  #[512,160]
  var12976=tf.sigmoid(var12975)
  #[512,160]
  var12977=tf.matmul(var12963, var12446)
  #[512,160]
  var12978=tf.reshape(var12977, [512,160])
  #[512,160]
  var12979=tf.add(var12978, var12450)
  #[512,160]
  var12980=tf.tanh(var12979)
  #[512,160]
  var12981=tf.multiply(var12976, var12980)
  #[512,160]
  var12982=tf.add(var12972, var12981)
  #[512,160]
  var12983=tf.tanh(var12982)
  #[512,160]
  var12984=tf.multiply(var12967, var12983)
  #[512,160]
  var12985=tf.multiply(var12391, var12984)
  #[512,160]
  var12986=tf.reshape(var12985, [512,160])
  #[512,2]
  var12987=tf.matmul(var12986, var12459)
  #[512,2]
  var12988=tf.reshape(var12987, [512,2])
  #[512,2]
  var12989=tf.add(var12988, var12463)
  #[512,1,2]
  var12990=tf.reshape(var12989, [512,1,2])
  #[512,160]
  var12991=tf.multiply(var12395, var12984)
  #[512,1]
  var12992=var12416[:,16:17]
  #[512]
  var12993=tf.reshape(var12992, [512])
  #[512,12]
  var12994=tf.gather(params=var12415, indices=var12993, batch_dims=0, axis=0)
  #[512,12]
  var12995=tf.multiply(var12414, var12994)
  #[512,12]
  var12996=tf.multiply(var12410, var12995)
  #[512,172]
  var12997=tf.concat([var12991,var12996], axis=1)
  #[512,172]
  var12998=tf.reshape(var12997, [512,172])
  #[512,160]
  var12999=tf.matmul(var12998, var12424)
  #[512,160]
  var13000=tf.reshape(var12999, [512,160])
  #[512,160]
  var13001=tf.add(var13000, var12428)
  #[512,160]
  var13002=tf.sigmoid(var13001)
  #[512,160]
  var13003=tf.matmul(var12998, var12431)
  #[512,160]
  var13004=tf.reshape(var13003, [512,160])
  #[512,160]
  var13005=tf.add(var13004, var12435)
  #[512,160]
  var13006=tf.sigmoid(var13005)
  #[512,160]
  var13007=tf.multiply(var13006, var12982)
  #[512,160]
  var13008=tf.matmul(var12998, var12439)
  #[512,160]
  var13009=tf.reshape(var13008, [512,160])
  #[512,160]
  var13010=tf.add(var13009, var12443)
  #[512,160]
  var13011=tf.sigmoid(var13010)
  #[512,160]
  var13012=tf.matmul(var12998, var12446)
  #[512,160]
  var13013=tf.reshape(var13012, [512,160])
  #[512,160]
  var13014=tf.add(var13013, var12450)
  #[512,160]
  var13015=tf.tanh(var13014)
  #[512,160]
  var13016=tf.multiply(var13011, var13015)
  #[512,160]
  var13017=tf.add(var13007, var13016)
  #[512,160]
  var13018=tf.tanh(var13017)
  #[512,160]
  var13019=tf.multiply(var13002, var13018)
  #[512,160]
  var13020=tf.multiply(var12391, var13019)
  #[512,160]
  var13021=tf.reshape(var13020, [512,160])
  #[512,2]
  var13022=tf.matmul(var13021, var12459)
  #[512,2]
  var13023=tf.reshape(var13022, [512,2])
  #[512,2]
  var13024=tf.add(var13023, var12463)
  #[512,1,2]
  var13025=tf.reshape(var13024, [512,1,2])
  #[512,160]
  var13026=tf.multiply(var12395, var13019)
  #[512,1]
  var13027=var12416[:,17:18]
  #[512]
  var13028=tf.reshape(var13027, [512])
  #[512,12]
  var13029=tf.gather(params=var12415, indices=var13028, batch_dims=0, axis=0)
  #[512,12]
  var13030=tf.multiply(var12414, var13029)
  #[512,12]
  var13031=tf.multiply(var12410, var13030)
  #[512,172]
  var13032=tf.concat([var13026,var13031], axis=1)
  #[512,172]
  var13033=tf.reshape(var13032, [512,172])
  #[512,160]
  var13034=tf.matmul(var13033, var12424)
  #[512,160]
  var13035=tf.reshape(var13034, [512,160])
  #[512,160]
  var13036=tf.add(var13035, var12428)
  #[512,160]
  var13037=tf.sigmoid(var13036)
  #[512,160]
  var13038=tf.matmul(var13033, var12431)
  #[512,160]
  var13039=tf.reshape(var13038, [512,160])
  #[512,160]
  var13040=tf.add(var13039, var12435)
  #[512,160]
  var13041=tf.sigmoid(var13040)
  #[512,160]
  var13042=tf.multiply(var13041, var13017)
  #[512,160]
  var13043=tf.matmul(var13033, var12439)
  #[512,160]
  var13044=tf.reshape(var13043, [512,160])
  #[512,160]
  var13045=tf.add(var13044, var12443)
  #[512,160]
  var13046=tf.sigmoid(var13045)
  #[512,160]
  var13047=tf.matmul(var13033, var12446)
  #[512,160]
  var13048=tf.reshape(var13047, [512,160])
  #[512,160]
  var13049=tf.add(var13048, var12450)
  #[512,160]
  var13050=tf.tanh(var13049)
  #[512,160]
  var13051=tf.multiply(var13046, var13050)
  #[512,160]
  var13052=tf.add(var13042, var13051)
  #[512,160]
  var13053=tf.tanh(var13052)
  #[512,160]
  var13054=tf.multiply(var13037, var13053)
  #[512,160]
  var13055=tf.multiply(var12391, var13054)
  #[512,160]
  var13056=tf.reshape(var13055, [512,160])
  #[512,2]
  var13057=tf.matmul(var13056, var12459)
  #[512,2]
  var13058=tf.reshape(var13057, [512,2])
  #[512,2]
  var13059=tf.add(var13058, var12463)
  #[512,1,2]
  var13060=tf.reshape(var13059, [512,1,2])
  #[512,160]
  var13061=tf.multiply(var12395, var13054)
  #[512,1]
  var13062=var12416[:,18:19]
  #[512]
  var13063=tf.reshape(var13062, [512])
  #[512,12]
  var13064=tf.gather(params=var12415, indices=var13063, batch_dims=0, axis=0)
  #[512,12]
  var13065=tf.multiply(var12414, var13064)
  #[512,12]
  var13066=tf.multiply(var12410, var13065)
  #[512,172]
  var13067=tf.concat([var13061,var13066], axis=1)
  #[512,172]
  var13068=tf.reshape(var13067, [512,172])
  #[512,160]
  var13069=tf.matmul(var13068, var12424)
  #[512,160]
  var13070=tf.reshape(var13069, [512,160])
  #[512,160]
  var13071=tf.add(var13070, var12428)
  #[512,160]
  var13072=tf.sigmoid(var13071)
  #[512,160]
  var13073=tf.matmul(var13068, var12431)
  #[512,160]
  var13074=tf.reshape(var13073, [512,160])
  #[512,160]
  var13075=tf.add(var13074, var12435)
  #[512,160]
  var13076=tf.sigmoid(var13075)
  #[512,160]
  var13077=tf.multiply(var13076, var13052)
  #[512,160]
  var13078=tf.matmul(var13068, var12439)
  #[512,160]
  var13079=tf.reshape(var13078, [512,160])
  #[512,160]
  var13080=tf.add(var13079, var12443)
  #[512,160]
  var13081=tf.sigmoid(var13080)
  #[512,160]
  var13082=tf.matmul(var13068, var12446)
  #[512,160]
  var13083=tf.reshape(var13082, [512,160])
  #[512,160]
  var13084=tf.add(var13083, var12450)
  #[512,160]
  var13085=tf.tanh(var13084)
  #[512,160]
  var13086=tf.multiply(var13081, var13085)
  #[512,160]
  var13087=tf.add(var13077, var13086)
  #[512,160]
  var13088=tf.tanh(var13087)
  #[512,160]
  var13089=tf.multiply(var13072, var13088)
  #[512,160]
  var13090=tf.multiply(var12391, var13089)
  #[512,160]
  var13091=tf.reshape(var13090, [512,160])
  #[512,2]
  var13092=tf.matmul(var13091, var12459)
  #[512,2]
  var13093=tf.reshape(var13092, [512,2])
  #[512,2]
  var13094=tf.add(var13093, var12463)
  #[512,1,2]
  var13095=tf.reshape(var13094, [512,1,2])
  #[512,160]
  var13096=tf.multiply(var12395, var13089)
  #[512,1]
  var13097=var12416[:,19:20]
  #[512]
  var13098=tf.reshape(var13097, [512])
  #[512,12]
  var13099=tf.gather(params=var12415, indices=var13098, batch_dims=0, axis=0)
  #[512,12]
  var13100=tf.multiply(var12414, var13099)
  #[512,12]
  var13101=tf.multiply(var12410, var13100)
  #[512,172]
  var13102=tf.concat([var13096,var13101], axis=1)
  #[512,172]
  var13103=tf.reshape(var13102, [512,172])
  #[512,160]
  var13104=tf.matmul(var13103, var12424)
  #[512,160]
  var13105=tf.reshape(var13104, [512,160])
  #[512,160]
  var13106=tf.add(var13105, var12428)
  #[512,160]
  var13107=tf.sigmoid(var13106)
  #[512,160]
  var13108=tf.matmul(var13103, var12431)
  #[512,160]
  var13109=tf.reshape(var13108, [512,160])
  #[512,160]
  var13110=tf.add(var13109, var12435)
  #[512,160]
  var13111=tf.sigmoid(var13110)
  #[512,160]
  var13112=tf.multiply(var13111, var13087)
  #[512,160]
  var13113=tf.matmul(var13103, var12439)
  #[512,160]
  var13114=tf.reshape(var13113, [512,160])
  #[512,160]
  var13115=tf.add(var13114, var12443)
  #[512,160]
  var13116=tf.sigmoid(var13115)
  #[512,160]
  var13117=tf.matmul(var13103, var12446)
  #[512,160]
  var13118=tf.reshape(var13117, [512,160])
  #[512,160]
  var13119=tf.add(var13118, var12450)
  #[512,160]
  var13120=tf.tanh(var13119)
  #[512,160]
  var13121=tf.multiply(var13116, var13120)
  #[512,160]
  var13122=tf.add(var13112, var13121)
  #[512,160]
  var13123=tf.tanh(var13122)
  #[512,160]
  var13124=tf.multiply(var13107, var13123)
  #[512,160]
  var13125=tf.multiply(var12391, var13124)
  #[512,160]
  var13126=tf.reshape(var13125, [512,160])
  #[512,2]
  var13127=tf.matmul(var13126, var12459)
  #[512,2]
  var13128=tf.reshape(var13127, [512,2])
  #[512,2]
  var13129=tf.add(var13128, var12463)
  #[512,1,2]
  var13130=tf.reshape(var13129, [512,1,2])
  #[512,160]
  var13131=tf.multiply(var12395, var13124)
  #[512,1]
  var13132=var12416[:,20:21]
  #[512]
  var13133=tf.reshape(var13132, [512])
  #[512,12]
  var13134=tf.gather(params=var12415, indices=var13133, batch_dims=0, axis=0)
  #[512,12]
  var13135=tf.multiply(var12414, var13134)
  #[512,12]
  var13136=tf.multiply(var12410, var13135)
  #[512,172]
  var13137=tf.concat([var13131,var13136], axis=1)
  #[512,172]
  var13138=tf.reshape(var13137, [512,172])
  #[512,160]
  var13139=tf.matmul(var13138, var12424)
  #[512,160]
  var13140=tf.reshape(var13139, [512,160])
  #[512,160]
  var13141=tf.add(var13140, var12428)
  #[512,160]
  var13142=tf.sigmoid(var13141)
  #[512,160]
  var13143=tf.matmul(var13138, var12431)
  #[512,160]
  var13144=tf.reshape(var13143, [512,160])
  #[512,160]
  var13145=tf.add(var13144, var12435)
  #[512,160]
  var13146=tf.sigmoid(var13145)
  #[512,160]
  var13147=tf.multiply(var13146, var13122)
  #[512,160]
  var13148=tf.matmul(var13138, var12439)
  #[512,160]
  var13149=tf.reshape(var13148, [512,160])
  #[512,160]
  var13150=tf.add(var13149, var12443)
  #[512,160]
  var13151=tf.sigmoid(var13150)
  #[512,160]
  var13152=tf.matmul(var13138, var12446)
  #[512,160]
  var13153=tf.reshape(var13152, [512,160])
  #[512,160]
  var13154=tf.add(var13153, var12450)
  #[512,160]
  var13155=tf.tanh(var13154)
  #[512,160]
  var13156=tf.multiply(var13151, var13155)
  #[512,160]
  var13157=tf.add(var13147, var13156)
  #[512,160]
  var13158=tf.tanh(var13157)
  #[512,160]
  var13159=tf.multiply(var13142, var13158)
  #[512,160]
  var13160=tf.multiply(var12391, var13159)
  #[512,160]
  var13161=tf.reshape(var13160, [512,160])
  #[512,2]
  var13162=tf.matmul(var13161, var12459)
  #[512,2]
  var13163=tf.reshape(var13162, [512,2])
  #[512,2]
  var13164=tf.add(var13163, var12463)
  #[512,1,2]
  var13165=tf.reshape(var13164, [512,1,2])
  #[512,160]
  var13166=tf.multiply(var12395, var13159)
  #[512,1]
  var13167=var12416[:,21:22]
  #[512]
  var13168=tf.reshape(var13167, [512])
  #[512,12]
  var13169=tf.gather(params=var12415, indices=var13168, batch_dims=0, axis=0)
  #[512,12]
  var13170=tf.multiply(var12414, var13169)
  #[512,12]
  var13171=tf.multiply(var12410, var13170)
  #[512,172]
  var13172=tf.concat([var13166,var13171], axis=1)
  #[512,172]
  var13173=tf.reshape(var13172, [512,172])
  #[512,160]
  var13174=tf.matmul(var13173, var12424)
  #[512,160]
  var13175=tf.reshape(var13174, [512,160])
  #[512,160]
  var13176=tf.add(var13175, var12428)
  #[512,160]
  var13177=tf.sigmoid(var13176)
  #[512,160]
  var13178=tf.matmul(var13173, var12431)
  #[512,160]
  var13179=tf.reshape(var13178, [512,160])
  #[512,160]
  var13180=tf.add(var13179, var12435)
  #[512,160]
  var13181=tf.sigmoid(var13180)
  #[512,160]
  var13182=tf.multiply(var13181, var13157)
  #[512,160]
  var13183=tf.matmul(var13173, var12439)
  #[512,160]
  var13184=tf.reshape(var13183, [512,160])
  #[512,160]
  var13185=tf.add(var13184, var12443)
  #[512,160]
  var13186=tf.sigmoid(var13185)
  #[512,160]
  var13187=tf.matmul(var13173, var12446)
  #[512,160]
  var13188=tf.reshape(var13187, [512,160])
  #[512,160]
  var13189=tf.add(var13188, var12450)
  #[512,160]
  var13190=tf.tanh(var13189)
  #[512,160]
  var13191=tf.multiply(var13186, var13190)
  #[512,160]
  var13192=tf.add(var13182, var13191)
  #[512,160]
  var13193=tf.tanh(var13192)
  #[512,160]
  var13194=tf.multiply(var13177, var13193)
  #[512,160]
  var13195=tf.multiply(var12391, var13194)
  #[512,160]
  var13196=tf.reshape(var13195, [512,160])
  #[512,2]
  var13197=tf.matmul(var13196, var12459)
  #[512,2]
  var13198=tf.reshape(var13197, [512,2])
  #[512,2]
  var13199=tf.add(var13198, var12463)
  #[512,1,2]
  var13200=tf.reshape(var13199, [512,1,2])
  #[512,160]
  var13201=tf.multiply(var12395, var13194)
  #[512,1]
  var13202=var12416[:,22:23]
  #[512]
  var13203=tf.reshape(var13202, [512])
  #[512,12]
  var13204=tf.gather(params=var12415, indices=var13203, batch_dims=0, axis=0)
  #[512,12]
  var13205=tf.multiply(var12414, var13204)
  #[512,12]
  var13206=tf.multiply(var12410, var13205)
  #[512,172]
  var13207=tf.concat([var13201,var13206], axis=1)
  #[512,172]
  var13208=tf.reshape(var13207, [512,172])
  #[512,160]
  var13209=tf.matmul(var13208, var12424)
  #[512,160]
  var13210=tf.reshape(var13209, [512,160])
  #[512,160]
  var13211=tf.add(var13210, var12428)
  #[512,160]
  var13212=tf.sigmoid(var13211)
  #[512,160]
  var13213=tf.matmul(var13208, var12431)
  #[512,160]
  var13214=tf.reshape(var13213, [512,160])
  #[512,160]
  var13215=tf.add(var13214, var12435)
  #[512,160]
  var13216=tf.sigmoid(var13215)
  #[512,160]
  var13217=tf.multiply(var13216, var13192)
  #[512,160]
  var13218=tf.matmul(var13208, var12439)
  #[512,160]
  var13219=tf.reshape(var13218, [512,160])
  #[512,160]
  var13220=tf.add(var13219, var12443)
  #[512,160]
  var13221=tf.sigmoid(var13220)
  #[512,160]
  var13222=tf.matmul(var13208, var12446)
  #[512,160]
  var13223=tf.reshape(var13222, [512,160])
  #[512,160]
  var13224=tf.add(var13223, var12450)
  #[512,160]
  var13225=tf.tanh(var13224)
  #[512,160]
  var13226=tf.multiply(var13221, var13225)
  #[512,160]
  var13227=tf.add(var13217, var13226)
  #[512,160]
  var13228=tf.tanh(var13227)
  #[512,160]
  var13229=tf.multiply(var13212, var13228)
  #[512,160]
  var13230=tf.multiply(var12391, var13229)
  #[512,160]
  var13231=tf.reshape(var13230, [512,160])
  #[512,2]
  var13232=tf.matmul(var13231, var12459)
  #[512,2]
  var13233=tf.reshape(var13232, [512,2])
  #[512,2]
  var13234=tf.add(var13233, var12463)
  #[512,1,2]
  var13235=tf.reshape(var13234, [512,1,2])
  #[512,160]
  var13236=tf.multiply(var12395, var13229)
  #[512,1]
  var13237=var12416[:,23:24]
  #[512]
  var13238=tf.reshape(var13237, [512])
  #[512,12]
  var13239=tf.gather(params=var12415, indices=var13238, batch_dims=0, axis=0)
  #[512,12]
  var13240=tf.multiply(var12414, var13239)
  #[512,12]
  var13241=tf.multiply(var12410, var13240)
  #[512,172]
  var13242=tf.concat([var13236,var13241], axis=1)
  #[512,172]
  var13243=tf.reshape(var13242, [512,172])
  #[512,160]
  var13244=tf.matmul(var13243, var12424)
  #[512,160]
  var13245=tf.reshape(var13244, [512,160])
  #[512,160]
  var13246=tf.add(var13245, var12428)
  #[512,160]
  var13247=tf.sigmoid(var13246)
  #[512,160]
  var13248=tf.matmul(var13243, var12431)
  #[512,160]
  var13249=tf.reshape(var13248, [512,160])
  #[512,160]
  var13250=tf.add(var13249, var12435)
  #[512,160]
  var13251=tf.sigmoid(var13250)
  #[512,160]
  var13252=tf.multiply(var13251, var13227)
  #[512,160]
  var13253=tf.matmul(var13243, var12439)
  #[512,160]
  var13254=tf.reshape(var13253, [512,160])
  #[512,160]
  var13255=tf.add(var13254, var12443)
  #[512,160]
  var13256=tf.sigmoid(var13255)
  #[512,160]
  var13257=tf.matmul(var13243, var12446)
  #[512,160]
  var13258=tf.reshape(var13257, [512,160])
  #[512,160]
  var13259=tf.add(var13258, var12450)
  #[512,160]
  var13260=tf.tanh(var13259)
  #[512,160]
  var13261=tf.multiply(var13256, var13260)
  #[512,160]
  var13262=tf.add(var13252, var13261)
  #[512,160]
  var13263=tf.tanh(var13262)
  #[512,160]
  var13264=tf.multiply(var13247, var13263)
  #[512,160]
  var13265=tf.multiply(var12391, var13264)
  #[512,160]
  var13266=tf.reshape(var13265, [512,160])
  #[512,2]
  var13267=tf.matmul(var13266, var12459)
  #[512,2]
  var13268=tf.reshape(var13267, [512,2])
  #[512,2]
  var13269=tf.add(var13268, var12463)
  #[512,1,2]
  var13270=tf.reshape(var13269, [512,1,2])
  #[512,160]
  var13271=tf.multiply(var12395, var13264)
  #[512,1]
  var13272=var12416[:,24:25]
  #[512]
  var13273=tf.reshape(var13272, [512])
  #[512,12]
  var13274=tf.gather(params=var12415, indices=var13273, batch_dims=0, axis=0)
  #[512,12]
  var13275=tf.multiply(var12414, var13274)
  #[512,12]
  var13276=tf.multiply(var12410, var13275)
  #[512,172]
  var13277=tf.concat([var13271,var13276], axis=1)
  #[512,172]
  var13278=tf.reshape(var13277, [512,172])
  #[512,160]
  var13279=tf.matmul(var13278, var12424)
  #[512,160]
  var13280=tf.reshape(var13279, [512,160])
  #[512,160]
  var13281=tf.add(var13280, var12428)
  #[512,160]
  var13282=tf.sigmoid(var13281)
  #[512,160]
  var13283=tf.matmul(var13278, var12431)
  #[512,160]
  var13284=tf.reshape(var13283, [512,160])
  #[512,160]
  var13285=tf.add(var13284, var12435)
  #[512,160]
  var13286=tf.sigmoid(var13285)
  #[512,160]
  var13287=tf.multiply(var13286, var13262)
  #[512,160]
  var13288=tf.matmul(var13278, var12439)
  #[512,160]
  var13289=tf.reshape(var13288, [512,160])
  #[512,160]
  var13290=tf.add(var13289, var12443)
  #[512,160]
  var13291=tf.sigmoid(var13290)
  #[512,160]
  var13292=tf.matmul(var13278, var12446)
  #[512,160]
  var13293=tf.reshape(var13292, [512,160])
  #[512,160]
  var13294=tf.add(var13293, var12450)
  #[512,160]
  var13295=tf.tanh(var13294)
  #[512,160]
  var13296=tf.multiply(var13291, var13295)
  #[512,160]
  var13297=tf.add(var13287, var13296)
  #[512,160]
  var13298=tf.tanh(var13297)
  #[512,160]
  var13299=tf.multiply(var13282, var13298)
  #[512,160]
  var13300=tf.multiply(var12391, var13299)
  #[512,160]
  var13301=tf.reshape(var13300, [512,160])
  #[512,2]
  var13302=tf.matmul(var13301, var12459)
  #[512,2]
  var13303=tf.reshape(var13302, [512,2])
  #[512,2]
  var13304=tf.add(var13303, var12463)
  #[512,1,2]
  var13305=tf.reshape(var13304, [512,1,2])
  #[512,160]
  var13306=tf.multiply(var12395, var13299)
  #[512,1]
  var13307=var12416[:,25:26]
  #[512]
  var13308=tf.reshape(var13307, [512])
  #[512,12]
  var13309=tf.gather(params=var12415, indices=var13308, batch_dims=0, axis=0)
  #[512,12]
  var13310=tf.multiply(var12414, var13309)
  #[512,12]
  var13311=tf.multiply(var12410, var13310)
  #[512,172]
  var13312=tf.concat([var13306,var13311], axis=1)
  #[512,172]
  var13313=tf.reshape(var13312, [512,172])
  #[512,160]
  var13314=tf.matmul(var13313, var12424)
  #[512,160]
  var13315=tf.reshape(var13314, [512,160])
  #[512,160]
  var13316=tf.add(var13315, var12428)
  #[512,160]
  var13317=tf.sigmoid(var13316)
  #[512,160]
  var13318=tf.matmul(var13313, var12431)
  #[512,160]
  var13319=tf.reshape(var13318, [512,160])
  #[512,160]
  var13320=tf.add(var13319, var12435)
  #[512,160]
  var13321=tf.sigmoid(var13320)
  #[512,160]
  var13322=tf.multiply(var13321, var13297)
  #[512,160]
  var13323=tf.matmul(var13313, var12439)
  #[512,160]
  var13324=tf.reshape(var13323, [512,160])
  #[512,160]
  var13325=tf.add(var13324, var12443)
  #[512,160]
  var13326=tf.sigmoid(var13325)
  #[512,160]
  var13327=tf.matmul(var13313, var12446)
  #[512,160]
  var13328=tf.reshape(var13327, [512,160])
  #[512,160]
  var13329=tf.add(var13328, var12450)
  #[512,160]
  var13330=tf.tanh(var13329)
  #[512,160]
  var13331=tf.multiply(var13326, var13330)
  #[512,160]
  var13332=tf.add(var13322, var13331)
  #[512,160]
  var13333=tf.tanh(var13332)
  #[512,160]
  var13334=tf.multiply(var13317, var13333)
  #[512,160]
  var13335=tf.multiply(var12391, var13334)
  #[512,160]
  var13336=tf.reshape(var13335, [512,160])
  #[512,2]
  var13337=tf.matmul(var13336, var12459)
  #[512,2]
  var13338=tf.reshape(var13337, [512,2])
  #[512,2]
  var13339=tf.add(var13338, var12463)
  #[512,1,2]
  var13340=tf.reshape(var13339, [512,1,2])
  #[512,160]
  var13341=tf.multiply(var12395, var13334)
  #[512,1]
  var13342=var12416[:,26:27]
  #[512]
  var13343=tf.reshape(var13342, [512])
  #[512,12]
  var13344=tf.gather(params=var12415, indices=var13343, batch_dims=0, axis=0)
  #[512,12]
  var13345=tf.multiply(var12414, var13344)
  #[512,12]
  var13346=tf.multiply(var12410, var13345)
  #[512,172]
  var13347=tf.concat([var13341,var13346], axis=1)
  #[512,172]
  var13348=tf.reshape(var13347, [512,172])
  #[512,160]
  var13349=tf.matmul(var13348, var12424)
  #[512,160]
  var13350=tf.reshape(var13349, [512,160])
  #[512,160]
  var13351=tf.add(var13350, var12428)
  #[512,160]
  var13352=tf.sigmoid(var13351)
  #[512,160]
  var13353=tf.matmul(var13348, var12431)
  #[512,160]
  var13354=tf.reshape(var13353, [512,160])
  #[512,160]
  var13355=tf.add(var13354, var12435)
  #[512,160]
  var13356=tf.sigmoid(var13355)
  #[512,160]
  var13357=tf.multiply(var13356, var13332)
  #[512,160]
  var13358=tf.matmul(var13348, var12439)
  #[512,160]
  var13359=tf.reshape(var13358, [512,160])
  #[512,160]
  var13360=tf.add(var13359, var12443)
  #[512,160]
  var13361=tf.sigmoid(var13360)
  #[512,160]
  var13362=tf.matmul(var13348, var12446)
  #[512,160]
  var13363=tf.reshape(var13362, [512,160])
  #[512,160]
  var13364=tf.add(var13363, var12450)
  #[512,160]
  var13365=tf.tanh(var13364)
  #[512,160]
  var13366=tf.multiply(var13361, var13365)
  #[512,160]
  var13367=tf.add(var13357, var13366)
  #[512,160]
  var13368=tf.tanh(var13367)
  #[512,160]
  var13369=tf.multiply(var13352, var13368)
  #[512,160]
  var13370=tf.multiply(var12391, var13369)
  #[512,160]
  var13371=tf.reshape(var13370, [512,160])
  #[512,2]
  var13372=tf.matmul(var13371, var12459)
  #[512,2]
  var13373=tf.reshape(var13372, [512,2])
  #[512,2]
  var13374=tf.add(var13373, var12463)
  #[512,1,2]
  var13375=tf.reshape(var13374, [512,1,2])
  #[512,160]
  var13376=tf.multiply(var12395, var13369)
  #[512,1]
  var13377=var12416[:,27:28]
  #[512]
  var13378=tf.reshape(var13377, [512])
  #[512,12]
  var13379=tf.gather(params=var12415, indices=var13378, batch_dims=0, axis=0)
  #[512,12]
  var13380=tf.multiply(var12414, var13379)
  #[512,12]
  var13381=tf.multiply(var12410, var13380)
  #[512,172]
  var13382=tf.concat([var13376,var13381], axis=1)
  #[512,172]
  var13383=tf.reshape(var13382, [512,172])
  #[512,160]
  var13384=tf.matmul(var13383, var12424)
  #[512,160]
  var13385=tf.reshape(var13384, [512,160])
  #[512,160]
  var13386=tf.add(var13385, var12428)
  #[512,160]
  var13387=tf.sigmoid(var13386)
  #[512,160]
  var13388=tf.matmul(var13383, var12431)
  #[512,160]
  var13389=tf.reshape(var13388, [512,160])
  #[512,160]
  var13390=tf.add(var13389, var12435)
  #[512,160]
  var13391=tf.sigmoid(var13390)
  #[512,160]
  var13392=tf.multiply(var13391, var13367)
  #[512,160]
  var13393=tf.matmul(var13383, var12439)
  #[512,160]
  var13394=tf.reshape(var13393, [512,160])
  #[512,160]
  var13395=tf.add(var13394, var12443)
  #[512,160]
  var13396=tf.sigmoid(var13395)
  #[512,160]
  var13397=tf.matmul(var13383, var12446)
  #[512,160]
  var13398=tf.reshape(var13397, [512,160])
  #[512,160]
  var13399=tf.add(var13398, var12450)
  #[512,160]
  var13400=tf.tanh(var13399)
  #[512,160]
  var13401=tf.multiply(var13396, var13400)
  #[512,160]
  var13402=tf.add(var13392, var13401)
  #[512,160]
  var13403=tf.tanh(var13402)
  #[512,160]
  var13404=tf.multiply(var13387, var13403)
  #[512,160]
  var13405=tf.multiply(var12391, var13404)
  #[512,160]
  var13406=tf.reshape(var13405, [512,160])
  #[512,2]
  var13407=tf.matmul(var13406, var12459)
  #[512,2]
  var13408=tf.reshape(var13407, [512,2])
  #[512,2]
  var13409=tf.add(var13408, var12463)
  #[512,1,2]
  var13410=tf.reshape(var13409, [512,1,2])
  #[512,160]
  var13411=tf.multiply(var12395, var13404)
  #[512,1]
  var13412=var12416[:,28:29]
  #[512]
  var13413=tf.reshape(var13412, [512])
  #[512,12]
  var13414=tf.gather(params=var12415, indices=var13413, batch_dims=0, axis=0)
  #[512,12]
  var13415=tf.multiply(var12414, var13414)
  #[512,12]
  var13416=tf.multiply(var12410, var13415)
  #[512,172]
  var13417=tf.concat([var13411,var13416], axis=1)
  #[512,172]
  var13418=tf.reshape(var13417, [512,172])
  #[512,160]
  var13419=tf.matmul(var13418, var12424)
  #[512,160]
  var13420=tf.reshape(var13419, [512,160])
  #[512,160]
  var13421=tf.add(var13420, var12428)
  #[512,160]
  var13422=tf.sigmoid(var13421)
  #[512,160]
  var13423=tf.matmul(var13418, var12431)
  #[512,160]
  var13424=tf.reshape(var13423, [512,160])
  #[512,160]
  var13425=tf.add(var13424, var12435)
  #[512,160]
  var13426=tf.sigmoid(var13425)
  #[512,160]
  var13427=tf.multiply(var13426, var13402)
  #[512,160]
  var13428=tf.matmul(var13418, var12439)
  #[512,160]
  var13429=tf.reshape(var13428, [512,160])
  #[512,160]
  var13430=tf.add(var13429, var12443)
  #[512,160]
  var13431=tf.sigmoid(var13430)
  #[512,160]
  var13432=tf.matmul(var13418, var12446)
  #[512,160]
  var13433=tf.reshape(var13432, [512,160])
  #[512,160]
  var13434=tf.add(var13433, var12450)
  #[512,160]
  var13435=tf.tanh(var13434)
  #[512,160]
  var13436=tf.multiply(var13431, var13435)
  #[512,160]
  var13437=tf.add(var13427, var13436)
  #[512,160]
  var13438=tf.tanh(var13437)
  #[512,160]
  var13439=tf.multiply(var13422, var13438)
  #[512,160]
  var13440=tf.multiply(var12391, var13439)
  #[512,160]
  var13441=tf.reshape(var13440, [512,160])
  #[512,2]
  var13442=tf.matmul(var13441, var12459)
  #[512,2]
  var13443=tf.reshape(var13442, [512,2])
  #[512,2]
  var13444=tf.add(var13443, var12463)
  #[512,1,2]
  var13445=tf.reshape(var13444, [512,1,2])
  #[512,160]
  var13446=tf.multiply(var12395, var13439)
  #[512,1]
  var13447=var12416[:,29:30]
  #[512]
  var13448=tf.reshape(var13447, [512])
  #[512,12]
  var13449=tf.gather(params=var12415, indices=var13448, batch_dims=0, axis=0)
  #[512,12]
  var13450=tf.multiply(var12414, var13449)
  #[512,12]
  var13451=tf.multiply(var12410, var13450)
  #[512,172]
  var13452=tf.concat([var13446,var13451], axis=1)
  #[512,172]
  var13453=tf.reshape(var13452, [512,172])
  #[512,160]
  var13454=tf.matmul(var13453, var12424)
  #[512,160]
  var13455=tf.reshape(var13454, [512,160])
  #[512,160]
  var13456=tf.add(var13455, var12428)
  #[512,160]
  var13457=tf.sigmoid(var13456)
  #[512,160]
  var13458=tf.matmul(var13453, var12431)
  #[512,160]
  var13459=tf.reshape(var13458, [512,160])
  #[512,160]
  var13460=tf.add(var13459, var12435)
  #[512,160]
  var13461=tf.sigmoid(var13460)
  #[512,160]
  var13462=tf.multiply(var13461, var13437)
  #[512,160]
  var13463=tf.matmul(var13453, var12439)
  #[512,160]
  var13464=tf.reshape(var13463, [512,160])
  #[512,160]
  var13465=tf.add(var13464, var12443)
  #[512,160]
  var13466=tf.sigmoid(var13465)
  #[512,160]
  var13467=tf.matmul(var13453, var12446)
  #[512,160]
  var13468=tf.reshape(var13467, [512,160])
  #[512,160]
  var13469=tf.add(var13468, var12450)
  #[512,160]
  var13470=tf.tanh(var13469)
  #[512,160]
  var13471=tf.multiply(var13466, var13470)
  #[512,160]
  var13472=tf.add(var13462, var13471)
  #[512,160]
  var13473=tf.tanh(var13472)
  #[512,160]
  var13474=tf.multiply(var13457, var13473)
  #[512,160]
  var13475=tf.multiply(var12391, var13474)
  #[512,160]
  var13476=tf.reshape(var13475, [512,160])
  #[512,2]
  var13477=tf.matmul(var13476, var12459)
  #[512,2]
  var13478=tf.reshape(var13477, [512,2])
  #[512,2]
  var13479=tf.add(var13478, var12463)
  #[512,1,2]
  var13480=tf.reshape(var13479, [512,1,2])
  #[512,160]
  var13481=tf.multiply(var12395, var13474)
  #[512,1]
  var13482=var12416[:,30:31]
  #[512]
  var13483=tf.reshape(var13482, [512])
  #[512,12]
  var13484=tf.gather(params=var12415, indices=var13483, batch_dims=0, axis=0)
  #[512,12]
  var13485=tf.multiply(var12414, var13484)
  #[512,12]
  var13486=tf.multiply(var12410, var13485)
  #[512,172]
  var13487=tf.concat([var13481,var13486], axis=1)
  #[512,172]
  var13488=tf.reshape(var13487, [512,172])
  #[512,160]
  var13489=tf.matmul(var13488, var12424)
  #[512,160]
  var13490=tf.reshape(var13489, [512,160])
  #[512,160]
  var13491=tf.add(var13490, var12428)
  #[512,160]
  var13492=tf.sigmoid(var13491)
  #[512,160]
  var13493=tf.matmul(var13488, var12431)
  #[512,160]
  var13494=tf.reshape(var13493, [512,160])
  #[512,160]
  var13495=tf.add(var13494, var12435)
  #[512,160]
  var13496=tf.sigmoid(var13495)
  #[512,160]
  var13497=tf.multiply(var13496, var13472)
  #[512,160]
  var13498=tf.matmul(var13488, var12439)
  #[512,160]
  var13499=tf.reshape(var13498, [512,160])
  #[512,160]
  var13500=tf.add(var13499, var12443)
  #[512,160]
  var13501=tf.sigmoid(var13500)
  #[512,160]
  var13502=tf.matmul(var13488, var12446)
  #[512,160]
  var13503=tf.reshape(var13502, [512,160])
  #[512,160]
  var13504=tf.add(var13503, var12450)
  #[512,160]
  var13505=tf.tanh(var13504)
  #[512,160]
  var13506=tf.multiply(var13501, var13505)
  #[512,160]
  var13507=tf.add(var13497, var13506)
  #[512,160]
  var13508=tf.tanh(var13507)
  #[512,160]
  var13509=tf.multiply(var13492, var13508)
  #[512,160]
  var13510=tf.multiply(var12391, var13509)
  #[512,160]
  var13511=tf.reshape(var13510, [512,160])
  #[512,2]
  var13512=tf.matmul(var13511, var12459)
  #[512,2]
  var13513=tf.reshape(var13512, [512,2])
  #[512,2]
  var13514=tf.add(var13513, var12463)
  #[512,1,2]
  var13515=tf.reshape(var13514, [512,1,2])
  #[512,160]
  var13516=tf.multiply(var12395, var13509)
  #[512,1]
  var13517=var12416[:,31:32]
  #[512]
  var13518=tf.reshape(var13517, [512])
  #[512,12]
  var13519=tf.gather(params=var12415, indices=var13518, batch_dims=0, axis=0)
  #[512,12]
  var13520=tf.multiply(var12414, var13519)
  #[512,12]
  var13521=tf.multiply(var12410, var13520)
  #[512,172]
  var13522=tf.concat([var13516,var13521], axis=1)
  #[512,172]
  var13523=tf.reshape(var13522, [512,172])
  #[512,160]
  var13524=tf.matmul(var13523, var12424)
  #[512,160]
  var13525=tf.reshape(var13524, [512,160])
  #[512,160]
  var13526=tf.add(var13525, var12428)
  #[512,160]
  var13527=tf.sigmoid(var13526)
  #[512,160]
  var13528=tf.matmul(var13523, var12431)
  #[512,160]
  var13529=tf.reshape(var13528, [512,160])
  #[512,160]
  var13530=tf.add(var13529, var12435)
  #[512,160]
  var13531=tf.sigmoid(var13530)
  #[512,160]
  var13532=tf.multiply(var13531, var13507)
  #[512,160]
  var13533=tf.matmul(var13523, var12439)
  #[512,160]
  var13534=tf.reshape(var13533, [512,160])
  #[512,160]
  var13535=tf.add(var13534, var12443)
  #[512,160]
  var13536=tf.sigmoid(var13535)
  #[512,160]
  var13537=tf.matmul(var13523, var12446)
  #[512,160]
  var13538=tf.reshape(var13537, [512,160])
  #[512,160]
  var13539=tf.add(var13538, var12450)
  #[512,160]
  var13540=tf.tanh(var13539)
  #[512,160]
  var13541=tf.multiply(var13536, var13540)
  #[512,160]
  var13542=tf.add(var13532, var13541)
  #[512,160]
  var13543=tf.tanh(var13542)
  #[512,160]
  var13544=tf.multiply(var13527, var13543)
  #[512,160]
  var13545=tf.multiply(var12391, var13544)
  #[512,160]
  var13546=tf.reshape(var13545, [512,160])
  #[512,2]
  var13547=tf.matmul(var13546, var12459)
  #[512,2]
  var13548=tf.reshape(var13547, [512,2])
  #[512,2]
  var13549=tf.add(var13548, var12463)
  #[512,1,2]
  var13550=tf.reshape(var13549, [512,1,2])
  #[512,160]
  var13551=tf.multiply(var12395, var13544)
  #[512,1]
  var13552=var12416[:,32:33]
  #[512]
  var13553=tf.reshape(var13552, [512])
  #[512,12]
  var13554=tf.gather(params=var12415, indices=var13553, batch_dims=0, axis=0)
  #[512,12]
  var13555=tf.multiply(var12414, var13554)
  #[512,12]
  var13556=tf.multiply(var12410, var13555)
  #[512,172]
  var13557=tf.concat([var13551,var13556], axis=1)
  #[512,172]
  var13558=tf.reshape(var13557, [512,172])
  #[512,160]
  var13559=tf.matmul(var13558, var12424)
  #[512,160]
  var13560=tf.reshape(var13559, [512,160])
  #[512,160]
  var13561=tf.add(var13560, var12428)
  #[512,160]
  var13562=tf.sigmoid(var13561)
  #[512,160]
  var13563=tf.matmul(var13558, var12431)
  #[512,160]
  var13564=tf.reshape(var13563, [512,160])
  #[512,160]
  var13565=tf.add(var13564, var12435)
  #[512,160]
  var13566=tf.sigmoid(var13565)
  #[512,160]
  var13567=tf.multiply(var13566, var13542)
  #[512,160]
  var13568=tf.matmul(var13558, var12439)
  #[512,160]
  var13569=tf.reshape(var13568, [512,160])
  #[512,160]
  var13570=tf.add(var13569, var12443)
  #[512,160]
  var13571=tf.sigmoid(var13570)
  #[512,160]
  var13572=tf.matmul(var13558, var12446)
  #[512,160]
  var13573=tf.reshape(var13572, [512,160])
  #[512,160]
  var13574=tf.add(var13573, var12450)
  #[512,160]
  var13575=tf.tanh(var13574)
  #[512,160]
  var13576=tf.multiply(var13571, var13575)
  #[512,160]
  var13577=tf.add(var13567, var13576)
  #[512,160]
  var13578=tf.tanh(var13577)
  #[512,160]
  var13579=tf.multiply(var13562, var13578)
  #[512,160]
  var13580=tf.multiply(var12391, var13579)
  #[512,160]
  var13581=tf.reshape(var13580, [512,160])
  #[512,2]
  var13582=tf.matmul(var13581, var12459)
  #[512,2]
  var13583=tf.reshape(var13582, [512,2])
  #[512,2]
  var13584=tf.add(var13583, var12463)
  #[512,1,2]
  var13585=tf.reshape(var13584, [512,1,2])
  #[512,160]
  var13586=tf.multiply(var12395, var13579)
  #[512,1]
  var13587=var12416[:,33:34]
  #[512]
  var13588=tf.reshape(var13587, [512])
  #[512,12]
  var13589=tf.gather(params=var12415, indices=var13588, batch_dims=0, axis=0)
  #[512,12]
  var13590=tf.multiply(var12414, var13589)
  #[512,12]
  var13591=tf.multiply(var12410, var13590)
  #[512,172]
  var13592=tf.concat([var13586,var13591], axis=1)
  #[512,172]
  var13593=tf.reshape(var13592, [512,172])
  #[512,160]
  var13594=tf.matmul(var13593, var12424)
  #[512,160]
  var13595=tf.reshape(var13594, [512,160])
  #[512,160]
  var13596=tf.add(var13595, var12428)
  #[512,160]
  var13597=tf.sigmoid(var13596)
  #[512,160]
  var13598=tf.matmul(var13593, var12431)
  #[512,160]
  var13599=tf.reshape(var13598, [512,160])
  #[512,160]
  var13600=tf.add(var13599, var12435)
  #[512,160]
  var13601=tf.sigmoid(var13600)
  #[512,160]
  var13602=tf.multiply(var13601, var13577)
  #[512,160]
  var13603=tf.matmul(var13593, var12439)
  #[512,160]
  var13604=tf.reshape(var13603, [512,160])
  #[512,160]
  var13605=tf.add(var13604, var12443)
  #[512,160]
  var13606=tf.sigmoid(var13605)
  #[512,160]
  var13607=tf.matmul(var13593, var12446)
  #[512,160]
  var13608=tf.reshape(var13607, [512,160])
  #[512,160]
  var13609=tf.add(var13608, var12450)
  #[512,160]
  var13610=tf.tanh(var13609)
  #[512,160]
  var13611=tf.multiply(var13606, var13610)
  #[512,160]
  var13612=tf.add(var13602, var13611)
  #[512,160]
  var13613=tf.tanh(var13612)
  #[512,160]
  var13614=tf.multiply(var13597, var13613)
  #[512,160]
  var13615=tf.multiply(var12391, var13614)
  #[512,160]
  var13616=tf.reshape(var13615, [512,160])
  #[512,2]
  var13617=tf.matmul(var13616, var12459)
  #[512,2]
  var13618=tf.reshape(var13617, [512,2])
  #[512,2]
  var13619=tf.add(var13618, var12463)
  #[512,1,2]
  var13620=tf.reshape(var13619, [512,1,2])
  #[512,160]
  var13621=tf.multiply(var12395, var13614)
  #[512,1]
  var13622=var12416[:,34:35]
  #[512]
  var13623=tf.reshape(var13622, [512])
  #[512,12]
  var13624=tf.gather(params=var12415, indices=var13623, batch_dims=0, axis=0)
  #[512,12]
  var13625=tf.multiply(var12414, var13624)
  #[512,12]
  var13626=tf.multiply(var12410, var13625)
  #[512,172]
  var13627=tf.concat([var13621,var13626], axis=1)
  #[512,172]
  var13628=tf.reshape(var13627, [512,172])
  #[512,160]
  var13629=tf.matmul(var13628, var12424)
  #[512,160]
  var13630=tf.reshape(var13629, [512,160])
  #[512,160]
  var13631=tf.add(var13630, var12428)
  #[512,160]
  var13632=tf.sigmoid(var13631)
  #[512,160]
  var13633=tf.matmul(var13628, var12431)
  #[512,160]
  var13634=tf.reshape(var13633, [512,160])
  #[512,160]
  var13635=tf.add(var13634, var12435)
  #[512,160]
  var13636=tf.sigmoid(var13635)
  #[512,160]
  var13637=tf.multiply(var13636, var13612)
  #[512,160]
  var13638=tf.matmul(var13628, var12439)
  #[512,160]
  var13639=tf.reshape(var13638, [512,160])
  #[512,160]
  var13640=tf.add(var13639, var12443)
  #[512,160]
  var13641=tf.sigmoid(var13640)
  #[512,160]
  var13642=tf.matmul(var13628, var12446)
  #[512,160]
  var13643=tf.reshape(var13642, [512,160])
  #[512,160]
  var13644=tf.add(var13643, var12450)
  #[512,160]
  var13645=tf.tanh(var13644)
  #[512,160]
  var13646=tf.multiply(var13641, var13645)
  #[512,160]
  var13647=tf.add(var13637, var13646)
  #[512,160]
  var13648=tf.tanh(var13647)
  #[512,160]
  var13649=tf.multiply(var13632, var13648)
  #[512,160]
  var13650=tf.multiply(var12391, var13649)
  #[512,160]
  var13651=tf.reshape(var13650, [512,160])
  #[512,2]
  var13652=tf.matmul(var13651, var12459)
  #[512,2]
  var13653=tf.reshape(var13652, [512,2])
  #[512,2]
  var13654=tf.add(var13653, var12463)
  #[512,1,2]
  var13655=tf.reshape(var13654, [512,1,2])
  #[512,160]
  var13656=tf.multiply(var12395, var13649)
  #[512,1]
  var13657=var12416[:,35:36]
  #[512]
  var13658=tf.reshape(var13657, [512])
  #[512,12]
  var13659=tf.gather(params=var12415, indices=var13658, batch_dims=0, axis=0)
  #[512,12]
  var13660=tf.multiply(var12414, var13659)
  #[512,12]
  var13661=tf.multiply(var12410, var13660)
  #[512,172]
  var13662=tf.concat([var13656,var13661], axis=1)
  #[512,172]
  var13663=tf.reshape(var13662, [512,172])
  #[512,160]
  var13664=tf.matmul(var13663, var12424)
  #[512,160]
  var13665=tf.reshape(var13664, [512,160])
  #[512,160]
  var13666=tf.add(var13665, var12428)
  #[512,160]
  var13667=tf.sigmoid(var13666)
  #[512,160]
  var13668=tf.matmul(var13663, var12431)
  #[512,160]
  var13669=tf.reshape(var13668, [512,160])
  #[512,160]
  var13670=tf.add(var13669, var12435)
  #[512,160]
  var13671=tf.sigmoid(var13670)
  #[512,160]
  var13672=tf.multiply(var13671, var13647)
  #[512,160]
  var13673=tf.matmul(var13663, var12439)
  #[512,160]
  var13674=tf.reshape(var13673, [512,160])
  #[512,160]
  var13675=tf.add(var13674, var12443)
  #[512,160]
  var13676=tf.sigmoid(var13675)
  #[512,160]
  var13677=tf.matmul(var13663, var12446)
  #[512,160]
  var13678=tf.reshape(var13677, [512,160])
  #[512,160]
  var13679=tf.add(var13678, var12450)
  #[512,160]
  var13680=tf.tanh(var13679)
  #[512,160]
  var13681=tf.multiply(var13676, var13680)
  #[512,160]
  var13682=tf.add(var13672, var13681)
  #[512,160]
  var13683=tf.tanh(var13682)
  #[512,160]
  var13684=tf.multiply(var13667, var13683)
  #[512,160]
  var13685=tf.multiply(var12391, var13684)
  #[512,160]
  var13686=tf.reshape(var13685, [512,160])
  #[512,2]
  var13687=tf.matmul(var13686, var12459)
  #[512,2]
  var13688=tf.reshape(var13687, [512,2])
  #[512,2]
  var13689=tf.add(var13688, var12463)
  #[512,1,2]
  var13690=tf.reshape(var13689, [512,1,2])
  #[512,160]
  var13691=tf.multiply(var12395, var13684)
  #[512,1]
  var13692=var12416[:,36:37]
  #[512]
  var13693=tf.reshape(var13692, [512])
  #[512,12]
  var13694=tf.gather(params=var12415, indices=var13693, batch_dims=0, axis=0)
  #[512,12]
  var13695=tf.multiply(var12414, var13694)
  #[512,12]
  var13696=tf.multiply(var12410, var13695)
  #[512,172]
  var13697=tf.concat([var13691,var13696], axis=1)
  #[512,172]
  var13698=tf.reshape(var13697, [512,172])
  #[512,160]
  var13699=tf.matmul(var13698, var12424)
  #[512,160]
  var13700=tf.reshape(var13699, [512,160])
  #[512,160]
  var13701=tf.add(var13700, var12428)
  #[512,160]
  var13702=tf.sigmoid(var13701)
  #[512,160]
  var13703=tf.matmul(var13698, var12431)
  #[512,160]
  var13704=tf.reshape(var13703, [512,160])
  #[512,160]
  var13705=tf.add(var13704, var12435)
  #[512,160]
  var13706=tf.sigmoid(var13705)
  #[512,160]
  var13707=tf.multiply(var13706, var13682)
  #[512,160]
  var13708=tf.matmul(var13698, var12439)
  #[512,160]
  var13709=tf.reshape(var13708, [512,160])
  #[512,160]
  var13710=tf.add(var13709, var12443)
  #[512,160]
  var13711=tf.sigmoid(var13710)
  #[512,160]
  var13712=tf.matmul(var13698, var12446)
  #[512,160]
  var13713=tf.reshape(var13712, [512,160])
  #[512,160]
  var13714=tf.add(var13713, var12450)
  #[512,160]
  var13715=tf.tanh(var13714)
  #[512,160]
  var13716=tf.multiply(var13711, var13715)
  #[512,160]
  var13717=tf.add(var13707, var13716)
  #[512,160]
  var13718=tf.tanh(var13717)
  #[512,160]
  var13719=tf.multiply(var13702, var13718)
  #[512,160]
  var13720=tf.multiply(var12391, var13719)
  #[512,160]
  var13721=tf.reshape(var13720, [512,160])
  #[512,2]
  var13722=tf.matmul(var13721, var12459)
  #[512,2]
  var13723=tf.reshape(var13722, [512,2])
  #[512,2]
  var13724=tf.add(var13723, var12463)
  #[512,1,2]
  var13725=tf.reshape(var13724, [512,1,2])
  #[512,160]
  var13726=tf.multiply(var12395, var13719)
  #[512,1]
  var13727=var12416[:,37:38]
  #[512]
  var13728=tf.reshape(var13727, [512])
  #[512,12]
  var13729=tf.gather(params=var12415, indices=var13728, batch_dims=0, axis=0)
  #[512,12]
  var13730=tf.multiply(var12414, var13729)
  #[512,12]
  var13731=tf.multiply(var12410, var13730)
  #[512,172]
  var13732=tf.concat([var13726,var13731], axis=1)
  #[512,172]
  var13733=tf.reshape(var13732, [512,172])
  #[512,160]
  var13734=tf.matmul(var13733, var12424)
  #[512,160]
  var13735=tf.reshape(var13734, [512,160])
  #[512,160]
  var13736=tf.add(var13735, var12428)
  #[512,160]
  var13737=tf.sigmoid(var13736)
  #[512,160]
  var13738=tf.matmul(var13733, var12431)
  #[512,160]
  var13739=tf.reshape(var13738, [512,160])
  #[512,160]
  var13740=tf.add(var13739, var12435)
  #[512,160]
  var13741=tf.sigmoid(var13740)
  #[512,160]
  var13742=tf.multiply(var13741, var13717)
  #[512,160]
  var13743=tf.matmul(var13733, var12439)
  #[512,160]
  var13744=tf.reshape(var13743, [512,160])
  #[512,160]
  var13745=tf.add(var13744, var12443)
  #[512,160]
  var13746=tf.sigmoid(var13745)
  #[512,160]
  var13747=tf.matmul(var13733, var12446)
  #[512,160]
  var13748=tf.reshape(var13747, [512,160])
  #[512,160]
  var13749=tf.add(var13748, var12450)
  #[512,160]
  var13750=tf.tanh(var13749)
  #[512,160]
  var13751=tf.multiply(var13746, var13750)
  #[512,160]
  var13752=tf.add(var13742, var13751)
  #[512,160]
  var13753=tf.tanh(var13752)
  #[512,160]
  var13754=tf.multiply(var13737, var13753)
  #[512,160]
  var13755=tf.multiply(var12391, var13754)
  #[512,160]
  var13756=tf.reshape(var13755, [512,160])
  #[512,2]
  var13757=tf.matmul(var13756, var12459)
  #[512,2]
  var13758=tf.reshape(var13757, [512,2])
  #[512,2]
  var13759=tf.add(var13758, var12463)
  #[512,1,2]
  var13760=tf.reshape(var13759, [512,1,2])
  #[512,160]
  var13761=tf.multiply(var12395, var13754)
  #[512,1]
  var13762=var12416[:,38:39]
  #[512]
  var13763=tf.reshape(var13762, [512])
  #[512,12]
  var13764=tf.gather(params=var12415, indices=var13763, batch_dims=0, axis=0)
  #[512,12]
  var13765=tf.multiply(var12414, var13764)
  #[512,12]
  var13766=tf.multiply(var12410, var13765)
  #[512,172]
  var13767=tf.concat([var13761,var13766], axis=1)
  #[512,172]
  var13768=tf.reshape(var13767, [512,172])
  #[512,160]
  var13769=tf.matmul(var13768, var12424)
  #[512,160]
  var13770=tf.reshape(var13769, [512,160])
  #[512,160]
  var13771=tf.add(var13770, var12428)
  #[512,160]
  var13772=tf.sigmoid(var13771)
  #[512,160]
  var13773=tf.matmul(var13768, var12431)
  #[512,160]
  var13774=tf.reshape(var13773, [512,160])
  #[512,160]
  var13775=tf.add(var13774, var12435)
  #[512,160]
  var13776=tf.sigmoid(var13775)
  #[512,160]
  var13777=tf.multiply(var13776, var13752)
  #[512,160]
  var13778=tf.matmul(var13768, var12439)
  #[512,160]
  var13779=tf.reshape(var13778, [512,160])
  #[512,160]
  var13780=tf.add(var13779, var12443)
  #[512,160]
  var13781=tf.sigmoid(var13780)
  #[512,160]
  var13782=tf.matmul(var13768, var12446)
  #[512,160]
  var13783=tf.reshape(var13782, [512,160])
  #[512,160]
  var13784=tf.add(var13783, var12450)
  #[512,160]
  var13785=tf.tanh(var13784)
  #[512,160]
  var13786=tf.multiply(var13781, var13785)
  #[512,160]
  var13787=tf.add(var13777, var13786)
  #[512,160]
  var13788=tf.tanh(var13787)
  #[512,160]
  var13789=tf.multiply(var13772, var13788)
  #[512,160]
  var13790=tf.multiply(var12391, var13789)
  #[512,160]
  var13791=tf.reshape(var13790, [512,160])
  #[512,2]
  var13792=tf.matmul(var13791, var12459)
  #[512,2]
  var13793=tf.reshape(var13792, [512,2])
  #[512,2]
  var13794=tf.add(var13793, var12463)
  #[512,1,2]
  var13795=tf.reshape(var13794, [512,1,2])
  #[512,160]
  var13796=tf.multiply(var12395, var13789)
  #[512,1]
  var13797=var12416[:,39:40]
  #[512]
  var13798=tf.reshape(var13797, [512])
  #[512,12]
  var13799=tf.gather(params=var12415, indices=var13798, batch_dims=0, axis=0)
  #[512,12]
  var13800=tf.multiply(var12414, var13799)
  #[512,12]
  var13801=tf.multiply(var12410, var13800)
  #[512,172]
  var13802=tf.concat([var13796,var13801], axis=1)
  #[512,172]
  var13803=tf.reshape(var13802, [512,172])
  #[512,160]
  var13804=tf.matmul(var13803, var12424)
  #[512,160]
  var13805=tf.reshape(var13804, [512,160])
  #[512,160]
  var13806=tf.add(var13805, var12428)
  #[512,160]
  var13807=tf.sigmoid(var13806)
  #[512,160]
  var13808=tf.matmul(var13803, var12431)
  #[512,160]
  var13809=tf.reshape(var13808, [512,160])
  #[512,160]
  var13810=tf.add(var13809, var12435)
  #[512,160]
  var13811=tf.sigmoid(var13810)
  #[512,160]
  var13812=tf.multiply(var13811, var13787)
  #[512,160]
  var13813=tf.matmul(var13803, var12439)
  #[512,160]
  var13814=tf.reshape(var13813, [512,160])
  #[512,160]
  var13815=tf.add(var13814, var12443)
  #[512,160]
  var13816=tf.sigmoid(var13815)
  #[512,160]
  var13817=tf.matmul(var13803, var12446)
  #[512,160]
  var13818=tf.reshape(var13817, [512,160])
  #[512,160]
  var13819=tf.add(var13818, var12450)
  #[512,160]
  var13820=tf.tanh(var13819)
  #[512,160]
  var13821=tf.multiply(var13816, var13820)
  #[512,160]
  var13822=tf.add(var13812, var13821)
  #[512,160]
  var13823=tf.tanh(var13822)
  #[512,160]
  var13824=tf.multiply(var13807, var13823)
  #[512,160]
  var13825=tf.multiply(var12391, var13824)
  #[512,160]
  var13826=tf.reshape(var13825, [512,160])
  #[512,2]
  var13827=tf.matmul(var13826, var12459)
  #[512,2]
  var13828=tf.reshape(var13827, [512,2])
  #[512,2]
  var13829=tf.add(var13828, var12463)
  #[512,1,2]
  var13830=tf.reshape(var13829, [512,1,2])
  #[512,160]
  var13831=tf.multiply(var12395, var13824)
  #[512,1]
  var13832=var12416[:,40:41]
  #[512]
  var13833=tf.reshape(var13832, [512])
  #[512,12]
  var13834=tf.gather(params=var12415, indices=var13833, batch_dims=0, axis=0)
  #[512,12]
  var13835=tf.multiply(var12414, var13834)
  #[512,12]
  var13836=tf.multiply(var12410, var13835)
  #[512,172]
  var13837=tf.concat([var13831,var13836], axis=1)
  #[512,172]
  var13838=tf.reshape(var13837, [512,172])
  #[512,160]
  var13839=tf.matmul(var13838, var12424)
  #[512,160]
  var13840=tf.reshape(var13839, [512,160])
  #[512,160]
  var13841=tf.add(var13840, var12428)
  #[512,160]
  var13842=tf.sigmoid(var13841)
  #[512,160]
  var13843=tf.matmul(var13838, var12431)
  #[512,160]
  var13844=tf.reshape(var13843, [512,160])
  #[512,160]
  var13845=tf.add(var13844, var12435)
  #[512,160]
  var13846=tf.sigmoid(var13845)
  #[512,160]
  var13847=tf.multiply(var13846, var13822)
  #[512,160]
  var13848=tf.matmul(var13838, var12439)
  #[512,160]
  var13849=tf.reshape(var13848, [512,160])
  #[512,160]
  var13850=tf.add(var13849, var12443)
  #[512,160]
  var13851=tf.sigmoid(var13850)
  #[512,160]
  var13852=tf.matmul(var13838, var12446)
  #[512,160]
  var13853=tf.reshape(var13852, [512,160])
  #[512,160]
  var13854=tf.add(var13853, var12450)
  #[512,160]
  var13855=tf.tanh(var13854)
  #[512,160]
  var13856=tf.multiply(var13851, var13855)
  #[512,160]
  var13857=tf.add(var13847, var13856)
  #[512,160]
  var13858=tf.tanh(var13857)
  #[512,160]
  var13859=tf.multiply(var13842, var13858)
  #[512,160]
  var13860=tf.multiply(var12391, var13859)
  #[512,160]
  var13861=tf.reshape(var13860, [512,160])
  #[512,2]
  var13862=tf.matmul(var13861, var12459)
  #[512,2]
  var13863=tf.reshape(var13862, [512,2])
  #[512,2]
  var13864=tf.add(var13863, var12463)
  #[512,1,2]
  var13865=tf.reshape(var13864, [512,1,2])
  #[512,160]
  var13866=tf.multiply(var12395, var13859)
  #[512,1]
  var13867=var12416[:,41:42]
  #[512]
  var13868=tf.reshape(var13867, [512])
  #[512,12]
  var13869=tf.gather(params=var12415, indices=var13868, batch_dims=0, axis=0)
  #[512,12]
  var13870=tf.multiply(var12414, var13869)
  #[512,12]
  var13871=tf.multiply(var12410, var13870)
  #[512,172]
  var13872=tf.concat([var13866,var13871], axis=1)
  #[512,172]
  var13873=tf.reshape(var13872, [512,172])
  #[512,160]
  var13874=tf.matmul(var13873, var12424)
  #[512,160]
  var13875=tf.reshape(var13874, [512,160])
  #[512,160]
  var13876=tf.add(var13875, var12428)
  #[512,160]
  var13877=tf.sigmoid(var13876)
  #[512,160]
  var13878=tf.matmul(var13873, var12431)
  #[512,160]
  var13879=tf.reshape(var13878, [512,160])
  #[512,160]
  var13880=tf.add(var13879, var12435)
  #[512,160]
  var13881=tf.sigmoid(var13880)
  #[512,160]
  var13882=tf.multiply(var13881, var13857)
  #[512,160]
  var13883=tf.matmul(var13873, var12439)
  #[512,160]
  var13884=tf.reshape(var13883, [512,160])
  #[512,160]
  var13885=tf.add(var13884, var12443)
  #[512,160]
  var13886=tf.sigmoid(var13885)
  #[512,160]
  var13887=tf.matmul(var13873, var12446)
  #[512,160]
  var13888=tf.reshape(var13887, [512,160])
  #[512,160]
  var13889=tf.add(var13888, var12450)
  #[512,160]
  var13890=tf.tanh(var13889)
  #[512,160]
  var13891=tf.multiply(var13886, var13890)
  #[512,160]
  var13892=tf.add(var13882, var13891)
  #[512,160]
  var13893=tf.tanh(var13892)
  #[512,160]
  var13894=tf.multiply(var13877, var13893)
  #[512,160]
  var13895=tf.multiply(var12391, var13894)
  #[512,160]
  var13896=tf.reshape(var13895, [512,160])
  #[512,2]
  var13897=tf.matmul(var13896, var12459)
  #[512,2]
  var13898=tf.reshape(var13897, [512,2])
  #[512,2]
  var13899=tf.add(var13898, var12463)
  #[512,1,2]
  var13900=tf.reshape(var13899, [512,1,2])
  #[512,160]
  var13901=tf.multiply(var12395, var13894)
  #[512,1]
  var13902=var12416[:,42:43]
  #[512]
  var13903=tf.reshape(var13902, [512])
  #[512,12]
  var13904=tf.gather(params=var12415, indices=var13903, batch_dims=0, axis=0)
  #[512,12]
  var13905=tf.multiply(var12414, var13904)
  #[512,12]
  var13906=tf.multiply(var12410, var13905)
  #[512,172]
  var13907=tf.concat([var13901,var13906], axis=1)
  #[512,172]
  var13908=tf.reshape(var13907, [512,172])
  #[512,160]
  var13909=tf.matmul(var13908, var12424)
  #[512,160]
  var13910=tf.reshape(var13909, [512,160])
  #[512,160]
  var13911=tf.add(var13910, var12428)
  #[512,160]
  var13912=tf.sigmoid(var13911)
  #[512,160]
  var13913=tf.matmul(var13908, var12431)
  #[512,160]
  var13914=tf.reshape(var13913, [512,160])
  #[512,160]
  var13915=tf.add(var13914, var12435)
  #[512,160]
  var13916=tf.sigmoid(var13915)
  #[512,160]
  var13917=tf.multiply(var13916, var13892)
  #[512,160]
  var13918=tf.matmul(var13908, var12439)
  #[512,160]
  var13919=tf.reshape(var13918, [512,160])
  #[512,160]
  var13920=tf.add(var13919, var12443)
  #[512,160]
  var13921=tf.sigmoid(var13920)
  #[512,160]
  var13922=tf.matmul(var13908, var12446)
  #[512,160]
  var13923=tf.reshape(var13922, [512,160])
  #[512,160]
  var13924=tf.add(var13923, var12450)
  #[512,160]
  var13925=tf.tanh(var13924)
  #[512,160]
  var13926=tf.multiply(var13921, var13925)
  #[512,160]
  var13927=tf.add(var13917, var13926)
  #[512,160]
  var13928=tf.tanh(var13927)
  #[512,160]
  var13929=tf.multiply(var13912, var13928)
  #[512,160]
  var13930=tf.multiply(var12391, var13929)
  #[512,160]
  var13931=tf.reshape(var13930, [512,160])
  #[512,2]
  var13932=tf.matmul(var13931, var12459)
  #[512,2]
  var13933=tf.reshape(var13932, [512,2])
  #[512,2]
  var13934=tf.add(var13933, var12463)
  #[512,1,2]
  var13935=tf.reshape(var13934, [512,1,2])
  #[512,160]
  var13936=tf.multiply(var12395, var13929)
  #[512,1]
  var13937=var12416[:,43:44]
  #[512]
  var13938=tf.reshape(var13937, [512])
  #[512,12]
  var13939=tf.gather(params=var12415, indices=var13938, batch_dims=0, axis=0)
  #[512,12]
  var13940=tf.multiply(var12414, var13939)
  #[512,12]
  var13941=tf.multiply(var12410, var13940)
  #[512,172]
  var13942=tf.concat([var13936,var13941], axis=1)
  #[512,172]
  var13943=tf.reshape(var13942, [512,172])
  #[512,160]
  var13944=tf.matmul(var13943, var12424)
  #[512,160]
  var13945=tf.reshape(var13944, [512,160])
  #[512,160]
  var13946=tf.add(var13945, var12428)
  #[512,160]
  var13947=tf.sigmoid(var13946)
  #[512,160]
  var13948=tf.matmul(var13943, var12431)
  #[512,160]
  var13949=tf.reshape(var13948, [512,160])
  #[512,160]
  var13950=tf.add(var13949, var12435)
  #[512,160]
  var13951=tf.sigmoid(var13950)
  #[512,160]
  var13952=tf.multiply(var13951, var13927)
  #[512,160]
  var13953=tf.matmul(var13943, var12439)
  #[512,160]
  var13954=tf.reshape(var13953, [512,160])
  #[512,160]
  var13955=tf.add(var13954, var12443)
  #[512,160]
  var13956=tf.sigmoid(var13955)
  #[512,160]
  var13957=tf.matmul(var13943, var12446)
  #[512,160]
  var13958=tf.reshape(var13957, [512,160])
  #[512,160]
  var13959=tf.add(var13958, var12450)
  #[512,160]
  var13960=tf.tanh(var13959)
  #[512,160]
  var13961=tf.multiply(var13956, var13960)
  #[512,160]
  var13962=tf.add(var13952, var13961)
  #[512,160]
  var13963=tf.tanh(var13962)
  #[512,160]
  var13964=tf.multiply(var13947, var13963)
  #[512,160]
  var13965=tf.multiply(var12391, var13964)
  #[512,160]
  var13966=tf.reshape(var13965, [512,160])
  #[512,2]
  var13967=tf.matmul(var13966, var12459)
  #[512,2]
  var13968=tf.reshape(var13967, [512,2])
  #[512,2]
  var13969=tf.add(var13968, var12463)
  #[512,1,2]
  var13970=tf.reshape(var13969, [512,1,2])
  #[512,160]
  var13971=tf.multiply(var12395, var13964)
  #[512,1]
  var13972=var12416[:,44:45]
  #[512]
  var13973=tf.reshape(var13972, [512])
  #[512,12]
  var13974=tf.gather(params=var12415, indices=var13973, batch_dims=0, axis=0)
  #[512,12]
  var13975=tf.multiply(var12414, var13974)
  #[512,12]
  var13976=tf.multiply(var12410, var13975)
  #[512,172]
  var13977=tf.concat([var13971,var13976], axis=1)
  #[512,172]
  var13978=tf.reshape(var13977, [512,172])
  #[512,160]
  var13979=tf.matmul(var13978, var12424)
  #[512,160]
  var13980=tf.reshape(var13979, [512,160])
  #[512,160]
  var13981=tf.add(var13980, var12428)
  #[512,160]
  var13982=tf.sigmoid(var13981)
  #[512,160]
  var13983=tf.matmul(var13978, var12431)
  #[512,160]
  var13984=tf.reshape(var13983, [512,160])
  #[512,160]
  var13985=tf.add(var13984, var12435)
  #[512,160]
  var13986=tf.sigmoid(var13985)
  #[512,160]
  var13987=tf.multiply(var13986, var13962)
  #[512,160]
  var13988=tf.matmul(var13978, var12439)
  #[512,160]
  var13989=tf.reshape(var13988, [512,160])
  #[512,160]
  var13990=tf.add(var13989, var12443)
  #[512,160]
  var13991=tf.sigmoid(var13990)
  #[512,160]
  var13992=tf.matmul(var13978, var12446)
  #[512,160]
  var13993=tf.reshape(var13992, [512,160])
  #[512,160]
  var13994=tf.add(var13993, var12450)
  #[512,160]
  var13995=tf.tanh(var13994)
  #[512,160]
  var13996=tf.multiply(var13991, var13995)
  #[512,160]
  var13997=tf.add(var13987, var13996)
  #[512,160]
  var13998=tf.tanh(var13997)
  #[512,160]
  var13999=tf.multiply(var13982, var13998)
  #[512,160]
  var14000=tf.multiply(var12391, var13999)
  #[512,160]
  var14001=tf.reshape(var14000, [512,160])
  #[512,2]
  var14002=tf.matmul(var14001, var12459)
  #[512,2]
  var14003=tf.reshape(var14002, [512,2])
  #[512,2]
  var14004=tf.add(var14003, var12463)
  #[512,1,2]
  var14005=tf.reshape(var14004, [512,1,2])
  #[512,160]
  var14006=tf.multiply(var12395, var13999)
  #[512,1]
  var14007=var12416[:,45:46]
  #[512]
  var14008=tf.reshape(var14007, [512])
  #[512,12]
  var14009=tf.gather(params=var12415, indices=var14008, batch_dims=0, axis=0)
  #[512,12]
  var14010=tf.multiply(var12414, var14009)
  #[512,12]
  var14011=tf.multiply(var12410, var14010)
  #[512,172]
  var14012=tf.concat([var14006,var14011], axis=1)
  #[512,172]
  var14013=tf.reshape(var14012, [512,172])
  #[512,160]
  var14014=tf.matmul(var14013, var12424)
  #[512,160]
  var14015=tf.reshape(var14014, [512,160])
  #[512,160]
  var14016=tf.add(var14015, var12428)
  #[512,160]
  var14017=tf.sigmoid(var14016)
  #[512,160]
  var14018=tf.matmul(var14013, var12431)
  #[512,160]
  var14019=tf.reshape(var14018, [512,160])
  #[512,160]
  var14020=tf.add(var14019, var12435)
  #[512,160]
  var14021=tf.sigmoid(var14020)
  #[512,160]
  var14022=tf.multiply(var14021, var13997)
  #[512,160]
  var14023=tf.matmul(var14013, var12439)
  #[512,160]
  var14024=tf.reshape(var14023, [512,160])
  #[512,160]
  var14025=tf.add(var14024, var12443)
  #[512,160]
  var14026=tf.sigmoid(var14025)
  #[512,160]
  var14027=tf.matmul(var14013, var12446)
  #[512,160]
  var14028=tf.reshape(var14027, [512,160])
  #[512,160]
  var14029=tf.add(var14028, var12450)
  #[512,160]
  var14030=tf.tanh(var14029)
  #[512,160]
  var14031=tf.multiply(var14026, var14030)
  #[512,160]
  var14032=tf.add(var14022, var14031)
  #[512,160]
  var14033=tf.tanh(var14032)
  #[512,160]
  var14034=tf.multiply(var14017, var14033)
  #[512,160]
  var14035=tf.multiply(var12391, var14034)
  #[512,160]
  var14036=tf.reshape(var14035, [512,160])
  #[512,2]
  var14037=tf.matmul(var14036, var12459)
  #[512,2]
  var14038=tf.reshape(var14037, [512,2])
  #[512,2]
  var14039=tf.add(var14038, var12463)
  #[512,1,2]
  var14040=tf.reshape(var14039, [512,1,2])
  #[512,160]
  var14041=tf.multiply(var12395, var14034)
  #[512,1]
  var14042=var12416[:,46:47]
  #[512]
  var14043=tf.reshape(var14042, [512])
  #[512,12]
  var14044=tf.gather(params=var12415, indices=var14043, batch_dims=0, axis=0)
  #[512,12]
  var14045=tf.multiply(var12414, var14044)
  #[512,12]
  var14046=tf.multiply(var12410, var14045)
  #[512,172]
  var14047=tf.concat([var14041,var14046], axis=1)
  #[512,172]
  var14048=tf.reshape(var14047, [512,172])
  #[512,160]
  var14049=tf.matmul(var14048, var12424)
  #[512,160]
  var14050=tf.reshape(var14049, [512,160])
  #[512,160]
  var14051=tf.add(var14050, var12428)
  #[512,160]
  var14052=tf.sigmoid(var14051)
  #[512,160]
  var14053=tf.matmul(var14048, var12431)
  #[512,160]
  var14054=tf.reshape(var14053, [512,160])
  #[512,160]
  var14055=tf.add(var14054, var12435)
  #[512,160]
  var14056=tf.sigmoid(var14055)
  #[512,160]
  var14057=tf.multiply(var14056, var14032)
  #[512,160]
  var14058=tf.matmul(var14048, var12439)
  #[512,160]
  var14059=tf.reshape(var14058, [512,160])
  #[512,160]
  var14060=tf.add(var14059, var12443)
  #[512,160]
  var14061=tf.sigmoid(var14060)
  #[512,160]
  var14062=tf.matmul(var14048, var12446)
  #[512,160]
  var14063=tf.reshape(var14062, [512,160])
  #[512,160]
  var14064=tf.add(var14063, var12450)
  #[512,160]
  var14065=tf.tanh(var14064)
  #[512,160]
  var14066=tf.multiply(var14061, var14065)
  #[512,160]
  var14067=tf.add(var14057, var14066)
  #[512,160]
  var14068=tf.tanh(var14067)
  #[512,160]
  var14069=tf.multiply(var14052, var14068)
  #[512,160]
  var14070=tf.multiply(var12391, var14069)
  #[512,160]
  var14071=tf.reshape(var14070, [512,160])
  #[512,2]
  var14072=tf.matmul(var14071, var12459)
  #[512,2]
  var14073=tf.reshape(var14072, [512,2])
  #[512,2]
  var14074=tf.add(var14073, var12463)
  #[512,1,2]
  var14075=tf.reshape(var14074, [512,1,2])
  #[512,160]
  var14076=tf.multiply(var12395, var14069)
  #[512,1]
  var14077=var12416[:,47:48]
  #[512]
  var14078=tf.reshape(var14077, [512])
  #[512,12]
  var14079=tf.gather(params=var12415, indices=var14078, batch_dims=0, axis=0)
  #[512,12]
  var14080=tf.multiply(var12414, var14079)
  #[512,12]
  var14081=tf.multiply(var12410, var14080)
  #[512,172]
  var14082=tf.concat([var14076,var14081], axis=1)
  #[512,172]
  var14083=tf.reshape(var14082, [512,172])
  #[512,160]
  var14084=tf.matmul(var14083, var12424)
  #[512,160]
  var14085=tf.reshape(var14084, [512,160])
  #[512,160]
  var14086=tf.add(var14085, var12428)
  #[512,160]
  var14087=tf.sigmoid(var14086)
  #[512,160]
  var14088=tf.matmul(var14083, var12431)
  #[512,160]
  var14089=tf.reshape(var14088, [512,160])
  #[512,160]
  var14090=tf.add(var14089, var12435)
  #[512,160]
  var14091=tf.sigmoid(var14090)
  #[512,160]
  var14092=tf.multiply(var14091, var14067)
  #[512,160]
  var14093=tf.matmul(var14083, var12439)
  #[512,160]
  var14094=tf.reshape(var14093, [512,160])
  #[512,160]
  var14095=tf.add(var14094, var12443)
  #[512,160]
  var14096=tf.sigmoid(var14095)
  #[512,160]
  var14097=tf.matmul(var14083, var12446)
  #[512,160]
  var14098=tf.reshape(var14097, [512,160])
  #[512,160]
  var14099=tf.add(var14098, var12450)
  #[512,160]
  var14100=tf.tanh(var14099)
  #[512,160]
  var14101=tf.multiply(var14096, var14100)
  #[512,160]
  var14102=tf.add(var14092, var14101)
  #[512,160]
  var14103=tf.tanh(var14102)
  #[512,160]
  var14104=tf.multiply(var14087, var14103)
  #[512,160]
  var14105=tf.multiply(var12391, var14104)
  #[512,160]
  var14106=tf.reshape(var14105, [512,160])
  #[512,2]
  var14107=tf.matmul(var14106, var12459)
  #[512,2]
  var14108=tf.reshape(var14107, [512,2])
  #[512,2]
  var14109=tf.add(var14108, var12463)
  #[512,1,2]
  var14110=tf.reshape(var14109, [512,1,2])
  #[512,160]
  var14111=tf.multiply(var12395, var14104)
  #[512,1]
  var14112=var12416[:,48:49]
  #[512]
  var14113=tf.reshape(var14112, [512])
  #[512,12]
  var14114=tf.gather(params=var12415, indices=var14113, batch_dims=0, axis=0)
  #[512,12]
  var14115=tf.multiply(var12414, var14114)
  #[512,12]
  var14116=tf.multiply(var12410, var14115)
  #[512,172]
  var14117=tf.concat([var14111,var14116], axis=1)
  #[512,172]
  var14118=tf.reshape(var14117, [512,172])
  #[512,160]
  var14119=tf.matmul(var14118, var12424)
  #[512,160]
  var14120=tf.reshape(var14119, [512,160])
  #[512,160]
  var14121=tf.add(var14120, var12428)
  #[512,160]
  var14122=tf.sigmoid(var14121)
  #[512,160]
  var14123=tf.matmul(var14118, var12431)
  #[512,160]
  var14124=tf.reshape(var14123, [512,160])
  #[512,160]
  var14125=tf.add(var14124, var12435)
  #[512,160]
  var14126=tf.sigmoid(var14125)
  #[512,160]
  var14127=tf.multiply(var14126, var14102)
  #[512,160]
  var14128=tf.matmul(var14118, var12439)
  #[512,160]
  var14129=tf.reshape(var14128, [512,160])
  #[512,160]
  var14130=tf.add(var14129, var12443)
  #[512,160]
  var14131=tf.sigmoid(var14130)
  #[512,160]
  var14132=tf.matmul(var14118, var12446)
  #[512,160]
  var14133=tf.reshape(var14132, [512,160])
  #[512,160]
  var14134=tf.add(var14133, var12450)
  #[512,160]
  var14135=tf.tanh(var14134)
  #[512,160]
  var14136=tf.multiply(var14131, var14135)
  #[512,160]
  var14137=tf.add(var14127, var14136)
  #[512,160]
  var14138=tf.tanh(var14137)
  #[512,160]
  var14139=tf.multiply(var14122, var14138)
  #[512,160]
  var14140=tf.multiply(var12391, var14139)
  #[512,160]
  var14141=tf.reshape(var14140, [512,160])
  #[512,2]
  var14142=tf.matmul(var14141, var12459)
  #[512,2]
  var14143=tf.reshape(var14142, [512,2])
  #[512,2]
  var14144=tf.add(var14143, var12463)
  #[512,1,2]
  var14145=tf.reshape(var14144, [512,1,2])
  #[512,160]
  var14146=tf.multiply(var12395, var14139)
  #[512,1]
  var14147=var12416[:,49:50]
  #[512]
  var14148=tf.reshape(var14147, [512])
  #[512,12]
  var14149=tf.gather(params=var12415, indices=var14148, batch_dims=0, axis=0)
  #[512,12]
  var14150=tf.multiply(var12414, var14149)
  #[512,12]
  var14151=tf.multiply(var12410, var14150)
  #[512,172]
  var14152=tf.concat([var14146,var14151], axis=1)
  #[512,172]
  var14153=tf.reshape(var14152, [512,172])
  #[512,160]
  var14154=tf.matmul(var14153, var12424)
  #[512,160]
  var14155=tf.reshape(var14154, [512,160])
  #[512,160]
  var14156=tf.add(var14155, var12428)
  #[512,160]
  var14157=tf.sigmoid(var14156)
  #[512,160]
  var14158=tf.matmul(var14153, var12431)
  #[512,160]
  var14159=tf.reshape(var14158, [512,160])
  #[512,160]
  var14160=tf.add(var14159, var12435)
  #[512,160]
  var14161=tf.sigmoid(var14160)
  #[512,160]
  var14162=tf.multiply(var14161, var14137)
  #[512,160]
  var14163=tf.matmul(var14153, var12439)
  #[512,160]
  var14164=tf.reshape(var14163, [512,160])
  #[512,160]
  var14165=tf.add(var14164, var12443)
  #[512,160]
  var14166=tf.sigmoid(var14165)
  #[512,160]
  var14167=tf.matmul(var14153, var12446)
  #[512,160]
  var14168=tf.reshape(var14167, [512,160])
  #[512,160]
  var14169=tf.add(var14168, var12450)
  #[512,160]
  var14170=tf.tanh(var14169)
  #[512,160]
  var14171=tf.multiply(var14166, var14170)
  #[512,160]
  var14172=tf.add(var14162, var14171)
  #[512,160]
  var14173=tf.tanh(var14172)
  #[512,160]
  var14174=tf.multiply(var14157, var14173)
  #[512,160]
  var14175=tf.multiply(var12391, var14174)
  #[512,160]
  var14176=tf.reshape(var14175, [512,160])
  #[512,2]
  var14177=tf.matmul(var14176, var12459)
  #[512,2]
  var14178=tf.reshape(var14177, [512,2])
  #[512,2]
  var14179=tf.add(var14178, var12463)
  #[512,1,2]
  var14180=tf.reshape(var14179, [512,1,2])
  #[512,50,2]
  var14181=tf.concat([var12465
                     ,var12500
                     ,var12535
                     ,var12570
                     ,var12605
                     ,var12640
                     ,var12675
                     ,var12710
                     ,var12745
                     ,var12780
                     ,var12815
                     ,var12850
                     ,var12885
                     ,var12920
                     ,var12955
                     ,var12990
                     ,var13025
                     ,var13060
                     ,var13095
                     ,var13130
                     ,var13165
                     ,var13200
                     ,var13235
                     ,var13270
                     ,var13305
                     ,var13340
                     ,var13375
                     ,var13410
                     ,var13445
                     ,var13480
                     ,var13515
                     ,var13550
                     ,var13585
                     ,var13620
                     ,var13655
                     ,var13690
                     ,var13725
                     ,var13760
                     ,var13795
                     ,var13830
                     ,var13865
                     ,var13900
                     ,var13935
                     ,var13970
                     ,var14005
                     ,var14040
                     ,var14075
                     ,var14110
                     ,var14145
                     ,var14180],
                     axis=1)
  #[512]
  var14182=yIndex
  #[512,2]
  var14183=tf.gather(params=var14181, indices=var14182, batch_dims=1, axis=1)
  #[512]
  var14184=tf.nn.softmax_cross_entropy_with_logits(labels=var12378, logits=var14183)
  #[512]
  var14185=tf.reshape(var14184, [512])
  #[]
  var14186=tf.reduce_mean(var14185, axis=0)
  #[1]
  var14187=tf.broadcast_to(tf.reshape(var12396, [1]), [1])
  #[]
  var14188=tf.reshape(var14187, [])
  #[]
  var14189=tf.add(var14186, var14188)
  #[512]
  var14190=tf.argmax(var14183, axis=1, output_type=tf.int32)
  #[512]
  var14191=tf.argmax(var12378, axis=1, output_type=tf.int32)
  #[512]
  var14192=tf.equal(var14190, var14191)
  #[512]
  var14193=tf.cast(var14192, tf.float32)
  #[512]
  var14194=tf.reshape(var14193, [512])
  #[]
  var14195=tf.reduce_mean(var14194, axis=0)
  #[512,2]
  var14196=tf.reshape(var14183, [512,2])
  #[512,2]
  var14197=tf.nn.softmax(var14196, axis=1)
  #[512,2]
  var14198=tf.reshape(var14197, [512,2])
  return {"loss":var14189,"accuracy":var14195,"y_":var14198}
runModel = {"function":runModel_fn
           ,"batched":True
           ,"placeholders":{"x":{"shape":[512,50],"dtype":tf.int32}
                           ,"yIndex":{"shape":[512],"dtype":tf.int32}
                           ,"y":{"shape":[512],"dtype":tf.int32}}}