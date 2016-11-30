%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	Erlann, multiprocess based artificial neural network in erlang.
%	
%	newPercNet(N) - Create a new perceptron network with N neurons, 
%		returns list of pids where first is output
%	trainPercNet(TrainingSet, Neurons, WeightAlgorithm, BiasAlgorithm) - 
%		TrainingSet - {[Input1,...InputN], Output}
%		Neurons - List of pids to neurons
%		WeightAlgorithm(W,D,U,X) - The function used for training weight
%		BiasAlgorithm(B,D,U) - The function used for training bias
%		
%	new(N) - Create a new network with N neurons, returns list of pids
%	connect(OutPid, InPid) - Connect neuron with OutPid to neuron with InPid
%	stop(ListOfPids) - Stop neurons with pids in ListOfPids.
%	setBias(Pid, Bias) - Set bias for neuron with Pid
%	setWeight(Pid, Weight) - Set Weight for neuron with Pid
%
%	get(List, N) - Get item N in List (should be in another module)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-module(erlann).
-export([trainPercNet/1, newPercNet/1, neuron/1, new/1, stop/1, 
	setWeight/2, setBias/2, get/2, call/2, heavySide/2, sigmoid/2, signal/2]).

-record(neuron_st, {outpids, weight, bias, function, charge, counter}).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%newMultiPercNet(1, _) ->
	
%newMultiPercNet(Nl, Nn) ->
%	spawn_link(erlann, fun fun(Pid) -> Neurons = newPercNet(Nn) end)
%	[Hn|Tn] = Neurons,
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
trainPercNet({[], Neurons, _, _}) ->
	Neurons;
trainPercNet({[Ht|Tt], Neurons, WeightAlgorithm, linear}) ->
	[Hn|Tn] = Neurons,
	{X, D} = Ht,
	spawn_link(fun() -> signal(Tn, X) end),
	receive {signal, Y} -> 
		setPercWeight(Tn, Ht, Y, WeightAlgorithm),
		setBias(Hn, fun(B) -> 
			B + ((D-Y) * 0.1) end)
	end,
	trainPercNet({Tt, Neurons, WeightAlgorithm, linear});
trainPercNet({[Ht|Tt], Neurons, WeightAlgorithm, BiasAlgorithm}) ->
	[Hn|Tn] = Neurons,
	{X, D} = Ht,
	spawn_link(fun() -> signal(Tn, X) end),
	receive {signal, Y} -> 
		setPercWeight(Tn, Ht, Y, WeightAlgorithm),
		setBias(Hn, fun(B) -> 
			BiasAlgorithm(B,D,Y) end)
	end,
	trainPercNet({Tt, Neurons, WeightAlgorithm, BiasAlgorithm}).

	
setPercWeight([], {[], _}, _, _) ->
	[];
setPercWeight([Hn|Tn], {[X|Tx], D}, Y, linear) ->
	setWeight(Hn, fun(W) -> 
		W + ((D-Y) * 0.1 * X) end),
	setPercWeight(Tn, {Tx, D}, Y, linear);
setPercWeight([Hn|Tn], {[X|Tx], D}, Y, WeightAlgorithm) ->
	setWeight(Hn, fun(W) -> 
		WeightAlgorithm(W,D,Y,X) end),
	setPercWeight(Tn, {Tx, D}, Y, WeightAlgorithm).

newPercNet(_, []) ->
	[];
newPercNet(First, [_|Tn]) ->
	case Tn of 
		[] ->
			newPercNet(First, Tn);
		Tn ->
			[HTn|_] = Tn,
			
			setBias(HTn, fun(_X) -> 0 end),
			setWeight(HTn, fun(_X) -> 0 end),
	
			connect(HTn, First),
			newPercNet(First, Tn)
	end.
newPercNet(N) ->
	Neurons = new(N),
	[First|_] = Neurons,
	connect(First, self()),
	setFunction(First, fun(Signal, Bias) -> heavySide(Signal, Bias) end),
	call([First], {setCounter, N-1}),
	newPercNet(First, Neurons),
	Neurons.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

heavySide(Signal, Bias) ->
	if Signal + Bias > 0 ->
		1;
	true ->
		0
	end.

sigmoid(Signal, Bias) ->
	K = 100,
	1/(1+math:exp((-1) * K * (Signal + Bias))).
	
charge(St = #neuron_st{charge = dynamic}, Signal) ->
	receive 
		{signal, More} ->
			charge(St, Signal + More);
		_ ->
			charge(St, Signal)
	after 
		10 -> 
			Weight = St#neuron_st.weight,
			Function = St#neuron_st.function,
			Weight * Function(Signal, St#neuron_st.bias)
	end;
charge(St = #neuron_st{charge = static, counter = 1}, Signal) ->
	Weight = St#neuron_st.weight,
	Function = St#neuron_st.function,
	Weight * Function(Signal, St#neuron_st.bias);
charge(St = #neuron_st{charge = static}, Signal) ->
	receive 
		{signal, More} ->
			charge(St#neuron_st{counter = St#neuron_st.counter - 1}, Signal + More);
		_ ->
			charge(St, Signal)
	after 
		100 ->
			io:fwrite("Too few neurons for static charge\n"),
			Weight = St#neuron_st.weight,
			Function = St#neuron_st.function,
			Weight * Function(Signal, St#neuron_st.bias)
	end.	
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
	
neuron(St = #neuron_st{outpids = []}) ->
	receive 
		stop ->
			io:fwrite("~p stopped\n", [self()]);
		disconnect ->
			io:fwrite("~p not connected\n", [self()]),
			neuron(St);
		{connect, OutPid} ->
			io:fwrite("Connected ~p to ~p\n", [self(), OutPid]),
			neuron(St#neuron_st{outpids = [OutPid]});
		{setWeight, NewWeight} ->
			io:fwrite("Weight for ~p set to ~p\n", [self(), NewWeight(St#neuron_st.weight)]),
			neuron(St#neuron_st{weight = NewWeight(St#neuron_st.weight)});
		{setBias, NewBias} ->
			io:fwrite("Bias for ~p set to ~p\n", [self(), NewBias(St#neuron_st.bias)]),
			neuron(St#neuron_st{bias = NewBias(St#neuron_st.bias)});
		{setFunction, NewFunction} ->
			io:fwrite("Function for ~p set to ~p\n", [self(), NewFunction]),
			neuron(St#neuron_st{function = NewFunction});
		{setCounter, NewCounter} ->
			neuron(St#neuron_st{counter = NewCounter});
		{getWeight} ->
			io:fwrite("Weight for ~p:	~p\n", [self(), St#neuron_st.weight]),
			neuron(St);
		{getBias} ->
			io:fwrite("Bias for ~p:	~p\n", [self(), St#neuron_st.bias]),
			neuron(St);
		{signal, Signal} ->
			OutPut = charge(St, Signal),
			io:fwrite("Output: ~p\n", [OutPut]),
			neuron(St);
		Other ->
			io:fwrite("~p is not a valid neuronal input", [Other]),
			neuron(St)
	end;
neuron(St = #neuron_st{outpids = [OutPid|To]}) when (To == []) ->
	receive 
		stop ->
			io:fwrite("~p stopped!\n", [self()]);
		disconnect ->
			io:fwrite("Disconnected from ~p to ~p\n", [self(), OutPid]),
			neuron(St#neuron_st{outpids = []});
		{connect, _} ->
			io:fwrite("Neuron is already connected\n"),
			neuron(St);
		{setWeight, NewWeight} ->
			io:fwrite("Weight for ~p set to ~p\n", [self(), NewWeight(St#neuron_st.weight)]),
			neuron(St#neuron_st{weight = NewWeight(St#neuron_st.weight)});
		{setBias, NewBias} ->
			io:fwrite("Bias for ~p set to ~p\n", [self(), NewBias(St#neuron_st.bias)]),
			neuron(St#neuron_st{bias = NewBias(St#neuron_st.bias)});
		{setFunction, NewFunction} ->
			io:fwrite("Function for ~p set to ~p\n", [self(), NewFunction]),
			neuron(St#neuron_st{function = NewFunction});
		{setCounter, NewCounter} ->
			neuron(St#neuron_st{counter = NewCounter});
		{getWeight} ->
			io:fwrite("Weight for ~p:	~p\n", [self(), St#neuron_st.weight]),
			neuron(St);
		{getBias} ->
			io:fwrite("Bias for ~p:	~p\n", [self(), St#neuron_st.bias]),
			neuron(St);
		{signal, Signal} ->
			OutPut = charge(St, Signal),
			OutPid ! {signal, OutPut},
			neuron(St);
		Other ->
			io:fwrite("~p is not a valid neuronal input", [Other]),
			neuron(St)
	end.
	
new({N, PidsIn}) ->
	if 
		N > 0 ->
			PidsOut = lists:append([spawn_link(erlann, neuron, [#neuron_st{
				outpids = [], 
				weight = 1,
				bias = 0,
				function = fun(Signal, _) -> Signal end,
				charge = static,
				counter = 1
				}])], PidsIn),
			new({N-1, PidsOut});
		true ->
			PidsIn
	end;
new(N) ->
	if 
		N > 0 ->
			PidsIn = [],
			PidsOut = lists:append([spawn_link(erlann, neuron, [#neuron_st{
				outpids = [], 
				weight = 1,
				bias = 0,
				function = fun(Signal, _) -> Signal end,
				charge = static,
				counter = 1
				}])], PidsIn),
			new({N-1, PidsOut});
		true ->
			io:fwrite("Not a valid number\n")
	end.


stop([]) ->
	io:fwrite("Network stopped\n");
stop(Pids) ->
	[H|T] = Pids,
	H ! stop,
	stop(T).	
	
connect(OutPid, InPid) ->
	OutPid ! {connect, InPid}.
	
setWeight(Pid, WeightFunction) ->
	Pid ! {setWeight, WeightFunction}.

setBias(Pid, Bias) ->
	Pid ! {setBias, Bias}.
	
setFunction(Pid, Function) ->
	Pid ! {setFunction, Function}.

signal([], []) ->
	[];
signal([Hp|Tp], [Hs|Ts]) ->
	Hp ! {signal, Hs},
	signal(Tp, Ts).

call([], _) ->
	[];
call([Hp|Tp], Call) ->
	Hp ! Call,
	call(Tp, Call).
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
get([],_) ->
	io:fwrite("Reached end of list\n");
get(List,1) ->
	[H|_] = List,
	H;
get(List,N) ->
	[_|T] = List,
	get(T,N-1).
	