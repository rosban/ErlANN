%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	Erlann, multiprocess based artificial neural network in erlang.
%	
%	newTestNet(N) - Create a new test network with N neurons, 
%		returns list of pids
%	%%(not done) newPercNet(N) - Create a new perceptron network with N neurons, 
%		returns list of pids
%
%	new(N) - Create a new network with N neurons, returns list of pids
%	connect(OutPid, InPid) - Connect neuron with OutPid to neuron with InPid
%	stop(ListOfPids) - Stop neurons with pids in ListOfPids.
%	setBias(Pid, Bias) - Set bias for neron with Pid
%	setWeight(Pid, Weight) - Set Weight for neuron with Pid
%
%	get(List, N) - Get item N in List (should be in another module)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-module(erlann).
-export([newTestNet/1, newPercNet/1, neuron/3, new/1, stop/1, setWeight/2, setBias/2, get/2]).

newTestNet([]) ->
	[];
newTestNet([H|T]) ->
	
	setBias(H, 2),
	setWeight(H, 2),
	
	case T of	
		[] ->
			newTestNet(T);
		T ->
			[HT|_] = T,
			connect(H, HT),
			newTestNet(T)
	end;		
newTestNet(N) ->
	Neurons = new(N),
	newTestNet(Neurons),
	Neurons.	
	
newPercNet({_N, [H|_T]}) ->
	setBias(H, 2); 
newPercNet(N) ->
	Neurons = new(N),
	newPercNet(Neurons),
	Neurons.
		
neuron([], Weight, Bias) ->
	receive 
		stop ->
			io:fwrite("~p stopped\n", [self()]);
		{setWeight, NewBias} ->
			io:fwrite("Bias for ~p set to ~p\n", [self(), NewBias]),
			neuron([], Weight, NewBias);
		{setBias, NewWeight} ->
			io:fwrite("Weight for ~p set to ~p\n", [self(), NewWeight]),
			neuron([], NewWeight, Bias);
		{connect, OutPid} ->
			io:fwrite("Connected ~p to ~p\n", [OutPid,self()]),
			neuron([OutPid], Weight, Bias);
		Signal ->
			io:fwrite("Output: ~p\n", [Signal]),
			neuron([], Weight, Bias)
	end;
neuron([OutPid], Weight, Bias) ->
	receive 
		stop ->
			io:fwrite("~p stopped!\n", [self()]);
		{setWeight, NewBias} ->
			io:fwrite("Bias for ~p set to ~p\n", [self(), NewBias]),
			neuron([OutPid], Weight, NewBias);
		{setBias, NewWeight} ->
			io:fwrite("Weight for ~p set to ~p\n", [self(), NewWeight]),
			neuron([OutPid], NewWeight, Bias);
		{connect, _} ->
			io:fwrite("Neuron is already connected\n"),
			neuron([OutPid], Weight, Bias);
		Signal ->
			OutPid ! (Signal*Weight+Bias),
			neuron([OutPid], Weight, Bias)
	end.
	
new({N, PidsIn}) ->
	if 
		N > 0 ->
			%Length = length(PidsIn),
			%[H|_] = PidsIn,
			PidsOut = lists:append([spawn_link(erlann, neuron, [[], 0, 0])], PidsIn),
			new({N-1, PidsOut});
		true ->
			PidsIn
	end;
new(N) ->
	if 
		N > 0 ->
			PidsIn = [],
			PidsOut = lists:append([spawn_link(erlann, neuron, [[], 0, 0])], PidsIn),
			new({N-1, PidsOut});
		true ->
			io:fwrite("Not a valid number\n")
	end.

connect(OutPid, InPid) ->
	OutPid ! {connect, InPid}.
	
stop([]) ->
	io:fwrite("Network stopped\n");
stop(Pids) ->
	[H|T] = Pids,
	H ! stop,
	stop(T).
	
setWeight(Pid, Weight) ->
	Pid ! {setWeight, Weight}.
	
setBias(Pid, Bias) ->
	Pid ! {setBias, Bias}.
		
get([],_) ->
	io:fwrite("Reached end of list\n");
get(List,1) ->
	[H|_] = List,
	H;
get(List,N) ->
	[_|T] = List,
	get(T,N-1).
	