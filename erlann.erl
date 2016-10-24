%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	Erlann, multiprocess based artificial neural network in erlang.
%	
%	newPercNet(N) - Create a new perceptron network with N neurons, 
%		returns list of pids where first is output
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
-export([testPercNet/0, trainPercNet/1, newPercNet/1, neuron/1, new/1, stop/1, setWeight/2, setBias/2, get/2]).

% -record(trainingset, {input, outcome})

%trainPercNet([Hn|Tn], {[Hin], Outcomes}) ->
%	connect(Hn, self()),
%	trainPercNet({Neurons, #trainingset{inputs = Inputs, outcomes = Outcomes}}).
testPercNet() ->
	Neurons = newPercNet(3),
	TrainingSet = [
	{[1, 1], 1},
	{[1, 0], 0},
	{[0, 1], 0},
	{[0, 0], 0},
	{[1, 1], 1},
	{[1, 0], 0},
	{[0, 1], 0},
	{[0, 0], 0},
	{[1, 1], 1},
	{[1, 0], 0},
	{[0, 1], 0},
	{[1, 1], 1},
	{[1, 0], 0},
	{[0, 1], 0},
	{[0, 0], 0},
	{[1, 1], 1},
	{[1, 0], 0},
	{[0, 1], 0},
	{[1, 1], 1},
	{[1, 0], 0},
	{[1, 1], 1},
	{[1, 0], 0},
	{[0, 1], 0},
	{[0, 0], 0},
	{[1, 1], 1},
	{[1, 0], 0},
	{[0, 1], 0},
	{[1, 1], 1},
	{[1, 0], 0}
	%{[0,2,3,2,5,3,0,0,4,2], 0},
	%{[1,2,3,4,5,6,7,8,9,10], 1}, 
	%{[1,0,3,4,1,6,7,8,9,10], 0},
	%{[1,0,3,4,1,6,3,8,9,1], 0},
	%{[2,3,4,5,6,7,8,9,10,11], 1}
	],
	trainPercNet({Neurons, TrainingSet}),
	
	%[{X,_}|_] = TrainingSet,
	[_|Tn] = Neurons,
	spawn_link(fun() -> signal(Tn, [0,1]) end),
	receive {signal, Y} -> 
		io:fwrite("Y: ~p\n", [Y])
	end,
	call(Tn, {getWeight}).

trainPercNet({Neurons, []}) ->
	Neurons;
trainPercNet({Neurons, [Ht|Tt]}) ->
	[_|Tn] = Neurons,
	{X, _} = Ht,
	spawn_link(fun() -> signal(Tn, X) end),
	receive {signal, Y} -> 
		setPercWeight(Tn, Ht, Y)
	end,
	% io:fwrite("Y: ~p\n", [Y]),
	trainPercNet({Neurons, Tt}).

setPercWeight([], {[], _}, _) ->
	[];
setPercWeight([Hn|Tn], {[X|Tx], D}, Y) ->
	setWeight(Hn, fun(W) -> W + X*(D-Y) end),
	setPercWeight(Tn, {Tx, D}, Y).

newPercNet(_, []) ->
	[];
newPercNet(First, [_|Tn]) ->
	case Tn of 
		[] ->
			newPercNet(First, Tn);
		Tn ->
			[HTn|_] = Tn,
			
			setBias(HTn, 0),
			setWeight(HTn, fun(_X) -> 0 end),
	
			connect(HTn, First),
			newPercNet(First, Tn)
	end.
newPercNet(N) ->
	Neurons = new(N),
	[First|_] = Neurons,
	connect(First, self()),
	setFunction(First, fun(Signal, Bias) -> heavySide(Signal, Bias) end),
	call([First], {setN, N-1}),
	newPercNet(First, Neurons),
	Neurons.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

heavySide(Signal, Bias) ->
	if Signal + Bias > 0 ->
		1;
	true ->
		0
	end.

chargeDynamic(Signal, Weight, Bias, Function) ->
	receive 
		{signal, More} ->
			chargeDynamic(Signal + More, Weight, Bias, Function);
		_ ->
			chargeDynamic(Signal, Weight, Bias, Function)
	after 
		10 ->
			Weight*Function(Signal, Bias)
	end.

chargeStatic(Signal, Weight, Bias, Function, 1) ->
	Weight*Function(Signal, Bias);
chargeStatic(Signal, Weight, Bias, Function, N) ->
	%io:fwrite("~p\n", [N]),
	receive 
		{signal, More} ->
			chargeStatic(Signal + More, Weight, Bias, Function, N-1);
		_ ->
			chargeStatic(Signal, Weight, Bias, Function, N)
	after 
		100 ->
			io:fwrite("Too few neurons for static charge\n"),
			Weight*Function(Signal, Bias)
	end.	
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
	
neuron({Weight, Bias, Function, N}) ->
	receive 
		stop ->
			io:fwrite("~p stopped\n", [self()]);
		disconnect ->
			% io:fwrite("~p not connected\n", [self()]);
			neuron({Weight, Bias, Function, N});
		{connect, OutPid} ->
			% io:fwrite("Connected ~p to ~p\n", [self(), OutPid]),
			neuron({[OutPid], Weight, Bias, Function, N});
		{setWeight, NewWeight} ->
			% io:fwrite("Weight for ~p set to ~p\n", [self(), NewWeight]),
			neuron({NewWeight(Weight), Bias, Function, N});
		{setBias, NewBias} ->
			% io:fwrite("Bias for ~p set to ~p\n", [self(), NewBias]),
			neuron({Weight, NewBias, Function, N});
		{setFunction, NewFunction} ->
			% io:fwrite("Function for ~p set to ~p\n", [self(), NewFunction]),
			neuron({Weight, Bias, NewFunction, N});
		{setN, NewN} ->
			neuron({Weight, Bias, Function, NewN});
		{getWeight} ->
			io:fwrite("Weight for ~p:	~p\n", [self(), Weight]),
			neuron({Weight, Bias, Function, N});
		{signal, Signal} ->
			_OutPut = chargeStatic(Signal, Weight, Bias, Function, N),
			% io:fwrite("Output: ~p\n", [OutPut]),
			neuron({Weight, Bias, Function, N});
		_Other ->
			% io:fwrite("~p is not a valid neuronal input", [Other]),
			neuron({Weight, Bias, Function, N})
	end;
neuron({[OutPid], Weight, Bias, Function, N}) ->
	receive 
		stop ->
			io:fwrite("~p stopped!\n", [self()]);
		disconnect ->
			% io:fwrite("Disconnected from ~p to ~p\n", [self(), OutPid]),
			neuron({Weight, Bias, Function, N});
		{connect, _} ->
			% io:fwrite("Neuron is already connected\n"),
			neuron({[OutPid], Weight, Bias, Function, N});
		{setWeight, NewWeight} ->
			% io:fwrite("Weight for ~p set to ~p\n", [self(), NewWeight(Weight)]),
			neuron({[OutPid], NewWeight(Weight), Bias, Function, N});
		{setBias, NewBias} ->
			% io:fwrite("Bias for ~p set to ~p\n", [self(), NewBias]),
			neuron({[OutPid], Weight, NewBias, Function, N});
		{setFunction, NewFunction} ->
			% io:fwrite("Function for ~p set to ~p\n", [self(), NewFunction]),
			neuron({[OutPid], Weight, Bias, NewFunction, N});
		{setN, NewN} ->
			neuron({[OutPid], Weight, Bias, Function, NewN});
		{getWeight} ->
			io:fwrite("Weight for ~p:	~p\n", [self(), Weight]),
			neuron({[OutPid], Weight, Bias, Function, N});
		{signal, Signal} ->
			OutPut = chargeStatic(Signal, Weight, Bias, Function, N),
			OutPid ! {signal, OutPut},
			neuron({[OutPid], Weight, Bias, Function, N});
		_Other ->
			% io:fwrite("~p is not a valid neuronal input", [Other]),
			neuron({[OutPid], Weight, Bias, Function, N})
	end.
	
new({N, PidsIn}) ->
	if 
		N > 0 ->
			%Length = length(PidsIn),
			%[H|_] = PidsIn,
			PidsOut = lists:append([spawn_link(erlann, neuron, [{1, 0, 
				fun(Signal, _) -> Signal end, 1}])], PidsIn),
			new({N-1, PidsOut});
		true ->
			PidsIn
	end;
new(N) ->
	if 
		N > 0 ->
			PidsIn = [],
			PidsOut = lists:append([spawn_link(erlann, neuron, [{1, 0,
				fun(Signal, _) -> Signal end, 1}])], PidsIn),
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
	