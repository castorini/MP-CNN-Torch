function flexLookUpPoin(emb_vecs, poin_vecs)
  if emb_vecs == nil or poin_vecs == nil then
	error("Not good!")
  end
  local poinDim = poin_vecs:size(2)
  local vocaSize = emb_vecs:size(1)
  local Dim = emb_vecs:size(2)
  local featext = nn.Sequential()
  local lookupPara=nn.ParallelTable()   		
  lookupPara:add(nn.LookupTable(vocaSize,Dim)) --numberWords*D
  lookupPara:add(nn.LookupTable(53,poinDim)) --numberWords*DPos
  featext:add(lookupPara)
  featext:add(nn.JoinTable(2)) -- numberWords*(D+DPos)  
  -----------------Initialization----------------
  for i=1, vocaSize do
	if 1 == 2 then
		featext:get(1):get(1).weight[i] = torch.randn(Dim)			
	else 
		local emb = torch.Tensor(Dim):copy(emb_vecs[i])
		featext:get(1):get(1).weight[i] = emb			
	end			
  end  
  
  for i=1, poin_vecs:size(1) do
	----POIN parts!!	
	local emb2 = torch.Tensor(poinDim):copy(poin_vecs[i])
	featext:get(1):get(2).weight[i] = emb2
  end    	  
  ---------------------------CLONE!-------------
  local modelQ = nn.Sequential()
  if 1 == 2 then  
  	modelQ= featext:clone()
  else
  	modelQ= featext:clone('weight','bias','gradWeight','gradBias')
  end
  local deepM = nn.Sequential()
  paraQuery=nn.ParallelTable()
  paraQuery:add(modelQ)
  paraQuery:add(featext)			  		
  deepM:add(paraQuery)
  return deepM
end

function flexLookUp(emb_vecs)
  if emb_vecs == nil then
	error("Not good!")
  end
	
  local vocaSize = emb_vecs:size(1)
  local Dim = emb_vecs:size(2)
  local featext = nn.Sequential()  	
  featext:add(nn.LookupTable(vocaSize,Dim))
  -----------------Initialization----------------
  for i=1, vocaSize do
	if 1 == 2 then
		featext:get(1).weight[i] = torch.randn(Dim)			
	else 
		local emb = torch.Tensor(Dim):copy(emb_vecs[i])
		featext:get(1).weight[i] = emb
		----No POIN parts!!			
	end			
  end  
  ---------------------------CLONE!-------------
  local modelQ = nn.Sequential()
  if 1 == 2 then  
  	modelQ= featext:clone()
  else
  	modelQ= featext:clone('weight','bias','gradWeight','gradBias')
  end
  local deepM = nn.Sequential()
  paraQuery=nn.ParallelTable()
  paraQuery:add(modelQ)
  paraQuery:add(featext)			 
  deepM:add(paraQuery)	
  return deepM
end

function createModel(mdl, vocsize, Dsize, nout, KKw)
    -- define model to train
    local network = nn.Sequential()
    local featext = nn.Sequential()
    local classifier = nn.Sequential()

    local conCon1 = nn.Sequential()
    local conCon2 = nn.Sequential()
    local conCon3 = nn.Sequential()
    local conCon4 = nn.Sequential()

    local parallelConcat1 = nn.Concat(1)
    local parallelConcat2 = nn.Concat(1)
    local parallelConcat3 = nn.Concat(1)
    local parallelConcat4 = nn.Concat(1)
    local parallelConcat5 = nn.Concat(1)

    local D     = Dsize --opt.dimension
    local kW    = KKw --opt.kwidth
    local dW    = 1 -- opt.dwidth
    local noExtra = false
    local nhid1 = 250 --opt.nhid1 
    local nhid2 = 250 --opt.nhid2
    local NumFilter = D
    local pR = 2 --opt.pR
    local layers=1
	    
    if mdl == 'deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinearExDGpPoinPercpt' then
		dofile "PaddingReshape.lua"
		
		deepQuery=nn.Sequential()
   		D = Dsize 
		local incep1max = nn.Sequential()
		incep1max:add(nn.TemporalConvolution(D,NumFilter,1,dw))
		  if pR == 1 then
		  	incep1max:add(nn.PReLU())
		  else 
		  	incep1max:add(nn.Tanh())
		  end		  
		  incep1max:add(nn.Max(1))
		  incep1max:add(nn.Reshape(NumFilter,1))		  
		  local incep2max = nn.Sequential()
		  incep2max:add(nn.Max(1))
		  incep2max:add(nn.Reshape(NumFilter,1))			  
		  local combineDepth = nn.Concat(2)
		  combineDepth:add(incep1max)
		  combineDepth:add(incep2max)
		  
		  local ngram = kW                
		  for cc = 2, ngram do
		    local incepMax = nn.Sequential()
		    if not noExtra then
		    	incepMax:add(nn.TemporalConvolution(D,D,1,dw)) --set
		    	if pR == 1 then
					incepMax:add(nn.PReLU())
				else 
					incepMax:add(nn.Tanh())
				end
		    end
		    incepMax:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
		    if pR == 1 then
			  	incepMax:add(nn.PReLU())
			else 
			  	incepMax:add(nn.Tanh())
			end
		    incepMax:add(nn.Max(1))
		    incepMax:add(nn.Reshape(NumFilter,1))		    		    
		  	combineDepth:add(incepMax)		    
		  end  		  
		  		  
		  local incep1min = nn.Sequential()
		  incep1min:add(nn.TemporalConvolution(D,NumFilter,1,dw))
		  if pR == 1 then
			incep1min:add(nn.PReLU())
		  else 
			incep1min:add(nn.Tanh())
		  end		  
		  incep1min:add(nn.Min(1))
		  incep1min:add(nn.Reshape(NumFilter,1))		  
		  local incep2min = nn.Sequential()
		  incep2min:add(nn.Min(1))
		  incep2min:add(nn.Reshape(NumFilter,1))		  
		  combineDepth:add(incep1min)
		  combineDepth:add(incep2min)
		  
		  for cc = 2, ngram do		    
		    local incepMin = nn.Sequential()
		    if not noExtra then
				incepMin:add(nn.TemporalConvolution(D,D,1,dw)) --set
				if pR == 1 then
					incepMin:add(nn.PReLU())
				else 
					incepMin:add(nn.Tanh())
				end
		    end		  
		    incepMin:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
		    if pR == 1 then
			  	incepMin:add(nn.PReLU())
			else 
			  	incepMin:add(nn.Tanh())
			end
		    incepMin:add(nn.Min(1))
		    incepMin:add(nn.Reshape(NumFilter,1))		    		  	
		  	combineDepth:add(incepMin)		      		    
		  end  
		  
		  local incep1mean = nn.Sequential()
		  incep1mean:add(nn.TemporalConvolution(D,NumFilter,1,dw))
		  if pR == 1 then
			incep1mean:add(nn.PReLU())
		  else 
			incep1mean:add(nn.Tanh())
		  end
		  incep1mean:add(nn.Mean(1))
		  incep1mean:add(nn.Reshape(NumFilter,1))		    		  		  
		  local incep2mean = nn.Sequential()
		  incep2mean:add(nn.Mean(1))
		  incep2mean:add(nn.Reshape(NumFilter,1))		  
		  combineDepth:add(incep1mean)
		  combineDepth:add(incep2mean)		  
		  for cc = 2, ngram do
		    local incepMean = nn.Sequential()
		    if not noExtra then
		    	incepMean:add(nn.TemporalConvolution(D,D,1,dw)) --set
		    	if pR == 1 then
				 	incepMean:add(nn.PReLU())
				else 
				 	incepMean:add(nn.Tanh())
				end
		    end
		    incepMean:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
		    if pR == 1 then
			 	incepMean:add(nn.PReLU())
			else 
			 	incepMean:add(nn.Tanh())
			end
		    incepMean:add(nn.Mean(1))
		    incepMean:add(nn.Reshape(NumFilter,1))			    
		    combineDepth:add(incepMean)	
		  end  
		  
		  local conceptFNum = 20
		  for cc = 1, ngram do
		  	local perConcept = nn.Sequential()
		    perConcept:add(nn.PaddingReshape(2,2)) --set
		    perConcept:add(nn.SpatialConvolutionMM(1,conceptFNum,1,ngram)) --set
		    perConcept:add(nn.Max(2)) --set
		    if pR == 1 then
			 	perConcept:add(nn.PReLU())
			else 
			 	perConcept:add(nn.Tanh())
			end
			perConcept:add(nn.Transpose({1,2}))
		    combineDepth:add(perConcept)	
		  end
		  for cc = 1, ngram do
		  	local perConcept = nn.Sequential()
		    perConcept:add(nn.PaddingReshape(2,2)) --set
		    perConcept:add(nn.SpatialConvolutionMM(1,conceptFNum,1,ngram)) --set
		    perConcept:add(nn.Min(2)) --set
		    if pR == 1 then
			 	perConcept:add(nn.PReLU())
			else 
			 	perConcept:add(nn.Tanh())
			end
			perConcept:add(nn.Transpose({1,2}))
		    combineDepth:add(perConcept)	
		  end
		  
		  featext:add(combineDepth)		
		  local items = (ngram+1)*3  		
		  local separator = items+2*conceptFNum*ngram
		  local sepModel = 0 
		  if sepModel == 1 then  
			modelQ= featext:clone()
		  else
			modelQ= featext:clone('weight','bias','gradWeight','gradBias')
		  end
		  paraQuery=nn.ParallelTable()
		  paraQuery:add(modelQ)
          paraQuery:add(featext)			
          deepQuery:add(paraQuery) 
		  deepQuery:add(nn.JoinTable(2)) 
			
			d=nn.Concat(1) 
			for i=1,items do
  			if i <= items/3 then 					
	  			for j=1,items/3 do
	  				--if j == i then
						local connection = nn.Sequential()
						local minus=nn.Concat(2)
						local c1=nn.Sequential()
						local c2=nn.Sequential()
						c1:add(nn.Select(2,i)) -- == D, not D*1
						c1:add(nn.Reshape(NumFilter,1)) --D*1 here					
						c2:add(nn.Select(2,separator+j))					
						c2:add(nn.Reshape(NumFilter,1))
						minus:add(c1)
						minus:add(c2)
						connection:add(minus) -- D*2						
						local similarityC=nn.Concat(1) -- multi similarity criteria			
						local s1=nn.Sequential()
						s1:add(nn.SplitTable(2))
						s1:add(nn.PairwiseDistance(2)) -- scalar
						local s2=nn.Sequential()
						if 1 < 3 then
							s2:add(nn.SplitTable(2))
						else
							s2:add(nn.Transpose({1,2})) 
							s2:add(nn.SoftMax())
							s2:add(nn.SplitTable(1))										
						end						
						s2:add(nn.CsDis()) -- scalar
						local s3=nn.Sequential()
						s3:add(nn.SplitTable(2))
						s3:add(nn.CSubTable()) -- linear
						s3:add(nn.Abs()) -- linear						
						similarityC:add(s1)
						similarityC:add(s2)					
						similarityC:add(s3)
						connection:add(similarityC) -- scalar											
						d:add(connection)
						--end
					end
				elseif i <= 2*items/3 then				
					for j=1+items/3, 2*items/3 do
						--if j == i then
						local connection = nn.Sequential()
						local minus=nn.Concat(2)
						local c1=nn.Sequential()
						local c2=nn.Sequential()
						c1:add(nn.Select(2,i)) -- == NumFilter, not NumFilter*1
						c1:add(nn.Reshape(NumFilter,1)) --NumFilter*1 here
						c2:add(nn.Select(2,separator+j))
						c2:add(nn.Reshape(NumFilter,1))
						minus:add(c1)
						minus:add(c2)
						connection:add(minus) -- D*2						
						local similarityC=nn.Concat(1) -- multi similarity criteria			
						local s1=nn.Sequential()
						s1:add(nn.SplitTable(2))
						s1:add(nn.PairwiseDistance(2)) -- scalar
						local s2=nn.Sequential()			
						if 1 < 3 then
							s2:add(nn.SplitTable(2))
						else
							s2:add(nn.Transpose({1,2})) -- D*2 -> 2*D
							s2:add(nn.SoftMax())
							s2:add(nn.SplitTable(1))										
						end									
						s2:add(nn.CsDis()) -- scalar						
						local s3=nn.Sequential()
						s3:add(nn.SplitTable(2))
						s3:add(nn.CSubTable()) -- linear
						s3:add(nn.Abs()) -- linear						
						similarityC:add(s1)
						similarityC:add(s2)					
						similarityC:add(s3)
						connection:add(similarityC) -- scalar												
						d:add(connection)
						--end
					end
				else 
					for j=1+2*items/3, items do
						--if j == i then
						local connection = nn.Sequential()
						local minus=nn.Concat(2)
						local c1=nn.Sequential()
						local c2=nn.Sequential()
						c1:add(nn.Select(2,i)) -- == D, not D*1
						c1:add(nn.Reshape(NumFilter,1)) --D*1 here
						c2:add(nn.Select(2,separator+j))
						c2:add(nn.Reshape(NumFilter,1))
						minus:add(c1)
						minus:add(c2)
						connection:add(minus) -- D*2						
						local similarityC=nn.Concat(1) -- multi similarity criteria			
						local s1=nn.Sequential()
						s1:add(nn.SplitTable(2))
						s1:add(nn.PairwiseDistance(2)) -- scalar
						local s2=nn.Sequential()					
						if 1 < 3 then
							s2:add(nn.SplitTable(2))
						else
							s2:add(nn.Transpose({1,2})) -- D*2 -> 2*D
							s2:add(nn.SoftMax())
							s2:add(nn.SplitTable(1))										
						end							
						s2:add(nn.CsDis()) -- scalar
						local s3=nn.Sequential()
						s3:add(nn.SplitTable(2))
						s3:add(nn.CSubTable()) -- linear
						s3:add(nn.Abs()) -- linear						
						similarityC:add(s1)
						similarityC:add(s2)					
						similarityC:add(s3)					
						connection:add(similarityC) -- scalar											
						d:add(connection)
						--end
					end		
				end
			end	
				  				
			for i=1,NumFilter do
				for j=1,3 do 
					local connection = nn.Sequential()
					connection:add(nn.Select(1,i)) -- == 2items
					connection:add(nn.Reshape(2*separator,1)) --2items*1 here					
					local minus=nn.Concat(2)
					local c1=nn.Sequential()
					local c2=nn.Sequential()
					if j == 1 then 
						c1:add(nn.Narrow(1,1,ngram+1)) -- first half (items/3)*1
						c2:add(nn.Narrow(1,separator+1,ngram+1)) -- first half (items/3)*1
					elseif j == 2 then
						c1:add(nn.Narrow(1,ngram+2,ngram+1)) -- 
						c2:add(nn.Narrow(1,separator+ngram+2,ngram+1)) 
					else
						c1:add(nn.Narrow(1,2*(ngram+1)+1,ngram+1)) 
						c2:add(nn.Narrow(1,separator+2*(ngram+1)+1,ngram+1)) --each is ngram+1 portion (max or min or mean)
					end						
					
					minus:add(c1)
					minus:add(c2)
					connection:add(minus) -- (items/3)*2					
					local similarityC=nn.Concat(1) 	
					local s1=nn.Sequential()
					s1:add(nn.SplitTable(2))
					s1:add(nn.PairwiseDistance(2)) -- scalar
					local s2=nn.Sequential()					
					if 1 >= 2 then
						s2:add(nn.Transpose({1,2})) -- (items/3)*2 -> 2*(items/3)
						s2:add(nn.SoftMax()) --for softmax have to do transpose from (item/3)*2 -> 2*(item/3)
						s2:add(nn.SplitTable(1)) --softmax only works on row						
					else 																				
						s2:add(nn.SplitTable(2)) --(items/3)*2
					end
					s2:add(nn.CsDis()) -- scalar
					--local s3=nn.Sequential()
					--s3:add(nn.SplitTable(2))
					--s3:add(nn.CSubTable()) -- linear
					--s3:add(nn.Abs()) -- linear						
					similarityC:add(s1)
					similarityC:add(s2)					
					--similarityC:add(s3)
					connection:add(similarityC) -- scalar											
					d:add(connection)				
				end
			end			
			
			for i=items+1,separator do
	  		local connection = nn.Sequential()
				local minus=nn.Concat(2)
				local c1=nn.Sequential()
				local c2=nn.Sequential()
				c1:add(nn.Select(2,i)) -- == D, not D*1
				c1:add(nn.Reshape(NumFilter,1)) --D*1 here
				c2:add(nn.Select(2,separator+i))
				c2:add(nn.Reshape(NumFilter,1))
				minus:add(c1)
				minus:add(c2)
				connection:add(minus) -- D*2						
				local similarityC=nn.Concat(1) 			
				local s1=nn.Sequential()
				s1:add(nn.SplitTable(2))
				s1:add(nn.PairwiseDistance(2)) -- scalar
				local s2=nn.Sequential()					
				if 1 < 3 then
					s2:add(nn.SplitTable(2))
				else
					s2:add(nn.Transpose({1,2})) 
					s2:add(nn.SoftMax())
					s2:add(nn.SplitTable(1))										
				end							
				s2:add(nn.CsDis()) -- scalar
				local s3=nn.Sequential()
				s3:add(nn.SplitTable(2))
				s3:add(nn.CSubTable()) -- linear
				s3:add(nn.Abs()) -- linear						
				similarityC:add(s1)
				similarityC:add(s2)					
				similarityC:add(s3)					
				connection:add(similarityC) -- scalar											
				d:add(connection)		
			end
	  	
			deepQuery:add(d)	    
			return deepQuery	
		end
end

