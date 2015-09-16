local PaddingReshape, parent = torch.class('nn.PaddingReshape', 'nn.Module')

function PaddingReshape:__init(...)
   parent.__init(self)
   local arg = {...}

   self.size = torch.LongStorage()
   self.batchsize = torch.LongStorage()
   if torch.type(arg[#arg]) == 'boolean' then
      self.batchMode = arg[#arg]
      table.remove(arg, #arg)
   end
   local n = #arg
   if n == 1 and torch.typename(arg[1]) == 'torch.LongStorage' then
      self.size:resize(#arg[1]):copy(arg[1])
   else
      self.size:resize(n) 
      for i=1,n do --modifed index
         self.size[i] = arg[i] --modified shift
      end
   end

   self.nelement = 1
   self.batchsize:resize(#self.size+1)
   for i=1,#self.size do
      self.nelement = self.nelement * self.size[i]
      self.batchsize[i+1] = self.size[i]
   end
   
   -- only used for non-contiguous input or gradOutput
   self._input = torch.Tensor()
   self._gradOutput = torch.Tensor()
end

function PaddingReshape:updateOutput(input)
   if not input:isContiguous() then
      self._input:resizeAs(input)
      self._input:copy(input)
      input = self._input
   end
   
   argsYoshi = torch.LongStorage()
   local nsi = #input:size() --modified
   argsYoshi:resize(nsi+1) 
   argsYoshi[1] = 1
   for i=2,nsi+1 do --modifed index
   	argsYoshi[i] = input:size()[i-1] --modified shift
   end
   self.batchMode = false
      
   if (self.batchMode == false) or (
         (self.batchMode == nil) and 
         (input:nElement() == self.nelement and input:size(1) ~= 1)
      ) then
      self.output:view(input, argsYoshi) --modified
   else
      self.batchsize[1] = input:size(1)
      self.output:view(input, self.batchsize)
   end
   return self.output
end

function PaddingReshape:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput:resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end
   
   self.gradInput:viewAs(gradOutput, input)
   return self.gradInput
end


function PaddingReshape:__tostring__()
  return torch.type(self) .. '(Pad ' ..
      table.concat(self.size:totable(), 'x') .. ')'
end