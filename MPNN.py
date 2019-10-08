import numpy as np
import tensorflow as tf
import sys, time, warnings
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors  
from sklearn.metrics import mean_absolute_error

class Model(object):

    def __init__(self, n_node, dim_node, dim_edge, dim_atom, dim_y, dim_h=64, n_mpnn_step=5, dr=0.2, batch_size=20, lr=0.0001, useGPU=True):

        warnings.filterwarnings('ignore')
        tf.logging.set_verbosity(tf.logging.ERROR)
        rdBase.DisableLog('rdApp.error') 
        rdBase.DisableLog('rdApp.warning')

        self.n_node=n_node
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.dim_atom=dim_atom
        self.dim_y=dim_y

        self.dim_h=dim_h
        self.n_mpnn_step=n_mpnn_step
        self.dr=dr
        self.batch_size=batch_size
        self.lr=lr

        # variables
        self.G = tf.Graph()
        self.G.as_default()

        self.trn_flag = tf.placeholder(tf.bool)
        
        self.node = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.dim_node])
        self.edge = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.n_node, self.dim_edge])      
        self.proximity = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.n_node, 1])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])
        
        self.hidden_0, self.hidden_n = self._MP(self.batch_size, self.node, tf.concat([self.edge, self.proximity], 3), self.n_mpnn_step, self.dim_h)
        self.Y_pred = self._Readout(self.batch_size, self.node, self.hidden_0, self.hidden_n, self.dim_h * 4, self.dim_y, self.dr)
                 
        # session
        self.saver = tf.train.Saver()
        if useGPU:
            self.sess = tf.Session()
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0} )
            self.sess = tf.Session(config=config)


    def train(self, DV_trn, DE_trn, DP_trn, DY_trn, DV_val, DE_val, DP_val, DY_val, load_path=None, save_path=None):

        ## objective function
        cost_Y_total = tf.reduce_mean(tf.reduce_sum(tf.abs(self.Y - self.Y_pred), 1))
        cost_Y_indiv = [tf.reduce_mean(tf.abs(self.Y[:,yid:yid+1] - self.Y_pred[:,yid:yid+1])) for yid in range(self.dim_y)]

        vars_MP = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MP')
        vars_Y = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Y/'+str(yid)+'/readout') for yid in range(self.dim_y)]

        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == np.sum([len(vs) for vs in vars_Y]) + len(vars_MP)
        
        train_op_total = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost_Y_total)
        train_op_indiv = [tf.train.AdamOptimizer(learning_rate=self.lr * 0.1).minimize(cost_Y_indiv[yid], var_list=vars_Y[yid]) for yid in range(self.dim_y)] 
                
        self.sess.run(tf.initializers.global_variables())            
        np.set_printoptions(precision=5, suppress=True)

        n_batch = int(len(DV_trn)/self.batch_size)
        n_batch_val = int(len(DV_val)/self.batch_size)

        if load_path is not None:
            self.saver.restore(self.sess, load_path)
            
        ## tranining
        max_epoch=500
        print('::: training')
        for yid in range(-1, self.dim_y):
        
            trn_log = np.zeros(max_epoch)
            val_t = np.zeros(max_epoch)
            for epoch in range(max_epoch):
    
                # training
                [DV_trn, DE_trn, DP_trn, DY_trn] = self._permutation([DV_trn, DE_trn, DP_trn, DY_trn])
                
                trnscores = np.zeros(n_batch) 
                if epoch > 0:
                    for i in range(n_batch):
        
                        start_=i*self.batch_size
                        end_=start_+self.batch_size
                        
                        assert self.batch_size == end_ - start_
                        
                        if yid==-1:
                            trnresult = self.sess.run([train_op_total, cost_Y_total],
                                                      feed_dict = {self.node: DV_trn[start_:end_], self.edge: DE_trn[start_:end_], 
                                                                   self.proximity: DP_trn[start_:end_], self.Y: DY_trn[start_:end_], self.trn_flag: True}) 
                        else:
                            trnresult = self.sess.run([train_op_indiv[yid], cost_Y_indiv[yid]],
                                                      feed_dict = {self.node: DV_trn[start_:end_], self.edge: DE_trn[start_:end_], 
                                                                   self.proximity: DP_trn[start_:end_], self.Y: DY_trn[start_:end_], self.trn_flag: True}) 
                            
                        trnscores[i] = trnresult[1]
                    
                    trn_log[epoch] = np.mean(trnscores)        
                    print('--training yid: ', yid, ' epoch id: ', epoch, ' trn log: ', trn_log[epoch])
                
                # validation
                DY_val_hat = self.test(DV_val, DE_val, DP_val)
                val_mse = np.array([mean_absolute_error(DY_val[:,yid:yid+1], DY_val_hat[:,yid:yid+1]) for yid in range(self.dim_y)])
                val_t[epoch] = np.sum(val_mse)
                print('--evaluation yid: ', yid, ' epoch id: ', epoch, ' val MAE: ', val_t[epoch], 'BEST: ', np.min(val_t[0:epoch+1]), np.min(val_t[0:epoch+1])/self.dim_y)
                print('--evaluation yid: ', yid, ' list: ', val_mse)         
              
                if epoch > 20 and np.min(val_t[0:epoch-20]) < np.min(val_t[epoch-20:epoch+1]):
                    print('--termination condition is met')
                    break
                
                elif np.min(val_t[0:epoch+1]) == val_t[epoch]:
                    self.saver.save(self.sess, save_path) 


    def test(self, DV_tst, DE_tst, DP_tst):
    
        n_batch_tst = int(len(DV_tst)/self.batch_size)
        DY_tst_hat=[]
        for i in range(n_batch_tst):
        
            start_=i*self.batch_size
            end_=start_+self.batch_size
            
            assert self.batch_size == end_ - start_
            
            DY_tst_batch = self.sess.run(self.Y_pred,
                                         feed_dict = {self.node: DV_tst[start_:end_], self.edge: DE_tst[start_:end_],
                                                      self.proximity: DP_tst[start_:end_], self.trn_flag: False})
            
            DY_tst_hat.append(DY_tst_batch)
        
        DY_tst_hat = np.concatenate(DY_tst_hat, 0)

        return DY_tst_hat      


    def _permutation(self, set):
    
        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]
    
        return set
        
    
    def _MP(self, batch_size, node, edge, n_step, hiddendim):

        def _embed_node(inp):
        
            inp = tf.reshape(inp, [batch_size * self.n_node, int(inp.shape[2])])
            inp = tf.layers.dense(inp, hiddendim * 4, activation = tf.nn.tanh)
            inp = tf.layers.dense(inp, hiddendim, activation = tf.nn.tanh)
        
            inp = tf.reshape(inp, [batch_size, self.n_node, hiddendim])
            inp = inp * mask
        
            return inp

        def _edge_nn(inp):
        
            inp = tf.reshape(inp, [batch_size * self.n_node * self.n_node, int(inp.shape[3])])
            inp = tf.layers.dense(inp, hiddendim * 4, activation = tf.nn.tanh)
            inp = tf.layers.dense(inp, hiddendim * hiddendim)
        
            inp = tf.reshape(inp, [batch_size, self.n_node, self.n_node, hiddendim, hiddendim])
            inp = inp * tf.reshape(1-tf.eye(self.n_node), [1, self.n_node, self.n_node, 1, 1])
            inp = inp * tf.reshape(mask, [batch_size, self.n_node, 1, 1, 1]) * tf.reshape(mask, [batch_size, 1, self.n_node, 1, 1])

            return inp

        def _MPNN(edge_wgt, node_hidden, n_step):
        
            def _msg_nn(wgt, node):
            
                wgt = tf.reshape(wgt, [batch_size * self.n_node, self.n_node * hiddendim, hiddendim])
                node = tf.reshape(node, [batch_size * self.n_node, hiddendim, 1])
            
                msg = tf.matmul(wgt, node)
                msg = tf.reshape(msg, [batch_size, self.n_node, self.n_node, hiddendim])
                msg = tf.transpose(msg, perm = [0, 2, 3, 1])
                msg = tf.reduce_mean(msg, 3)
            
                return msg

            def _update_GRU(msg, node, reuse_GRU):
            
                with tf.variable_scope('mpnn_gru', reuse=reuse_GRU):
            
                    msg = tf.reshape(msg, [batch_size * self.n_node, 1, hiddendim])
                    node = tf.reshape(node, [batch_size * self.n_node, hiddendim])
            
                    cell = tf.nn.rnn_cell.GRUCell(hiddendim)
                    _, node_next = tf.nn.dynamic_rnn(cell, msg, initial_state = node)
            
                    node_next = tf.reshape(node_next, [batch_size, self.n_node, hiddendim]) * mask
            
                return node_next

            nhs=[]
            for i in range(n_step):
                message_vec = _msg_nn(edge_wgt, node_hidden)
                node_hidden = _update_GRU(message_vec, node_hidden, reuse_GRU=(i!=0))
                nhs.append(node_hidden)
        
            out = tf.concat(nhs, axis=2)
            
            return out

        with tf.variable_scope('MP', reuse=False):
        
            mask = tf.reduce_max(node[:,:,:self.dim_atom], 2, keepdims=True)
            
            edge_wgt = _edge_nn(edge)
            hidden_0 = _embed_node(node)
            hidden_n = _MPNN(edge_wgt, hidden_0, n_step)
            
        return hidden_0, hidden_n


    def _Readout(self, batch_size, node, hidden_0, hidden_n, aggrdim, ydim, drate):
      
        def _readout(hidden_0, hidden_n, outdim):    
            
            def _attn_nn(inp, hdim):
            
                inp = tf.reshape(inp, [batch_size * self.n_node, int(inp.shape[2])])
                inp = tf.layers.dense(inp, hdim, activation = tf.nn.sigmoid)
                
                inp = tf.reshape(inp, [batch_size, self.n_node, hdim])
                 
                return inp
        
            def _tanh_nn(inp, hdim):
            
                inp = tf.reshape(inp, [batch_size * self.n_node, int(inp.shape[2])])
                inp = tf.layers.dense(inp, hdim)
            
                inp = tf.reshape(inp, [batch_size, self.n_node, hdim])
                
                return inp

            attn_wgt = _attn_nn(tf.concat([hidden_0, hidden_n], 2), aggrdim) 
            tanh_wgt = _tanh_nn(hidden_n, aggrdim)
            readout = tf.reduce_mean(tf.multiply(tanh_wgt, attn_wgt) * mask, 1)
            for _ in range(3):
                readout = tf.layers.dense(readout, aggrdim, activation = tf.nn.tanh)
                readout = tf.layers.dropout(readout, drate, training = self.trn_flag)
                
            pred = tf.layers.dense(readout, outdim) 
    
            return pred

        mask = tf.reduce_max(node[:,:,:self.dim_atom], 2, keepdims=True)

        rout = []
        for j in range(ydim):
        
            with tf.variable_scope('Y/'+str(j)+'/readout', reuse=False):
            
                readout = _readout(hidden_0, hidden_n, 1)	
                
            rout.append(readout)
        
        rout = tf.concat(rout, axis=1)

        return rout