require 'nn'
require 'optim'
require 'nnx'

cmd = torch.CmdLine()
cmd.argseparator = '_'
cmd:text()
cmd:text('Deep Ranking')
cmd:text()
cmd:text('Options:')
--cmd:option('-expid', 2030, 'experiment ID') --this is for ranking Jimmy's 1
cmd:option('-expid', 347, 'experiment ID') --this is for MSPrarpgrase
cmd:option('-plot', false, 'plot training and test errors, live')
cmd:option('-seed', 100, 'fixed input seed for repeatable experiments (-1=fresh seed')
--cmd:option('-network', '/scratch0/huah/deepQueryLogs/deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinear_exp_2014/nets/model.bestAcc', 'pre-trained model')
cmd:option('-network', '', 'pre-trained model')
cmd:option('-loadModel','','pretrained model')
cmd:option('-resOutput', true, 'Save results')
cmd:option('-sepModel', false, 'Separate the Training of Query and Doc Model')

-- Switch mode
cmd:option('-deepPara', 3, 'Switch to 1. Jimmy.QRanking or 2.deepPara (SemEvalSEME) or 3. MSPR or 4. SIC ')

-- data specific parameters
cmd:option('-loss', 'margin', 'loss function: mse | nll | margin | rank | kl')
cmd:option('-maxlossPara', 1, 'max^maxlossPara') -- on default 1. no more than 2.
cmd:option('-initialization', 'xavier', 'initialization method') -- kaiming/xavier/seld defined..

---this is for MSprarphrase data set! deep query 3.
--cmd:option('-addr', './MSparaphrase/', 'dataset addr')
cmd:option('-dtype', 'train.queryT.Pos.txt', 'dataset: train.data | dev.data | test.data')--withSemEval
cmd:option('-dever', 'dev.queryT.Pos.txt', 'dataset: train.data | dev.data | test.data')
--cmd:option('-tester', 'test.queryT.Pos.txt', 'dataset: train.data | dev.data | test.data')

--this is for deep query 1 task!
--cmd:option('-addr', './data/', 'dataset addr')
--cmd:option('-dtype', 'train.queryT.txt', 'dataset: train.data | dev.data | test.data')--withSemEval
--cmd:option('-dever', 'dev.queryT.txt', 'dataset: train.data | dev.data | test.data')
--cmd:option('-tester', 'test.queryT.txt', 'dataset: train.data | dev.data | test.data')

--For Deep Para 2 data sets
--cmd:option('-dtype', 'train.queryT.Pos.SingleDisFirst.txt', 'dataset: train.data | dev.data | test.data')
--cmd:option('-dever', 'dev.queryT.Pos.SingleDisFirst.txt', 'dataset: train.data | dev.data | test.data')
--cmd:option('-tester', 'test.queryT.Pos.txt', 'dataset: train.data | dev.data | test.data')

cmd:option('-full', true, 'use the entire training set or not: true | false')
cmd:option('-trsize', 5000, 'number of training samples in case not full')

-- set the model specific parameters
cmd:option('-model', 'deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinearExDGpPoinPercptNoGp', 'type of model to train')
cmd:option('-type', 'float', 'the model type: cuda | float')
cmd:option('-devid', 1, 'device id if using cuda')

cmd:option('-pR', 2, 'whether to use pRELU (choose 1) or tanh (choose 2)')
cmd:option('-normalizeCos', 1, 'whether normalize the input values for Cosine 1.no; 2.only second cos; 3.all cosine')

cmd:option('-dimension',300,'dimension of each vector in LookupTable')
cmd:option('-dimensionPos',200,'dimension of each Pos vector in PosLookupTable')
cmd:option('-numFilter',500,'dimension of number of output filter') -- have to be the same as dpos+d..
cmd:option('-conceptFNum',20,'number of per concept filter') -- per concept filters for building block B

cmd:option('-singleCt',4,'kernel width for temporal convolution')
cmd:option('-kwidth',3,'kernel width for temporal convolution')
cmd:option('-kwidthPos',2,'kernel width for Pos convolution')
cmd:option('-dwidth',1,'frame shift width for temporal convolution')
cmd:option('-nhid1', 250, 'number of units in first hidden layer')
cmd:option('-nhid2', 250, 'number of units in second hidden layer')
cmd:option('-saver', 1, 'Save the best model')

-- set the train hyper-parameters
cmd:option('-eta', 1e-2, 'learning rate of model at t=0')
cmd:option('-etadecay', 0, 'learning rate decay')
cmd:option('-weightdecay',0, 'weight decay (for SGD only)')
cmd:option('-momentum', 0.9, 'momentum (for SGD only)')
cmd:option('-lambdal1', 0, 'L1 penalty on the weights')
cmd:option('-lambdal2', 0, 'L2 penalty on the weights')

-- set the training parameters
cmd:option('-batch',10000,'random ranges for stochastic gradient descent')
cmd:option('-batchsize', 1, 'size of the mini-batch (1 for pure SGD)')
cmd:option('-threads', 1, 'max no of threads to be used')
cmd:option('-nepochs', 65, 'number of training epochs')
cmd:option('-regepoch', 200, 'number of epochs after which the regularization should start')
cmd:option('-statint', 1, 'number of iterations after which the model is to be saved')
cmd:option('-noembedding', false, 'Word embeddings is generated randomlym true or false')


cmd:text()
opt = cmd:parse(arg)

opt.statint = 10 --* opt.batchsize

opt.bestF1T = 0.0
opt.bestAccT = 0.0

opt.bestF1D = 0.0
opt.bestAccD = 0.0

-- the directory to save the log files
dname = "/scratch1/huah/deepQueryLogs"



-- the directory where the finetuned model will be stored
opt.bdir = paths.concat(dname,  opt.model .. '_exp_' .. opt.expid)
opt.mdir = paths.concat(dname,  opt.model .. '_exp_' .. opt.expid, 'nets')
opt.fdir = paths.concat(dname,  opt.model .. '_exp_' .. opt.expid, 'features')
-- create the directory if it does not exist
if not paths.filep(opt.bdir) then 
   os.execute('mkdir -p ' .. opt.bdir)
end
if not paths.filep(opt.mdir) then 
   os.execute('mkdir -p ' .. opt.mdir)
end
if not paths.filep(opt.fdir) then 
   os.execute('mkdir -p ' .. opt.fdir)
end
cmd:log(opt.mdir .. '/log', opt)


if opt.type == 'cuda' then 
   print('==> switching to cuda')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
   print(cutorch.getDeviceProperties(opt.devid))
end

