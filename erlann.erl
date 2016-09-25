-module(erlann).
-export([get/2, neuron/3, new/1, stop/1]).

get([],_) ->
	io:fwrite("Reached end of list!");
get(List,1) ->
	[H|_] = List,
	H;
get(List,N) ->
	[_|T] = List,
	get(T,N-1).
	
neuron([], Weight, Bias) ->
	receive stop ->
		io:fwrite("~p stopped!", [self()]);
	{connect, OutPid} ->
		io:fwrite("Connected ~p to ~p!", [OutPid,self()]),
		neuron([OutPid], Weight, Bias);
	Signal ->
		io:fwrite("Output: ~p\n", [Signal]),
		neuron([], Weight, Bias)
	end;
neuron([OutPid], Weight, Bias) ->
	receive stop ->
		io:fwrite("~p stopped!", [self()]);
	{connect, _} ->
		io:fwrite("Neuron is already connected!"),
		neuron([OutPid], Weight, Bias);
	Signal ->
		OutPid ! (Signal*Weight+Bias),
		neuron([OutPid], Weight, Bias)
	end.
	
new({N, PidsIn}) ->
	if N > 0 ->
		%Length = length(PidsIn),
		%[H|_] = PidsIn,
		PidsOut = lists:append([spawn(test, neuron, [[], 2, 2])], PidsIn),
		new({N-1, PidsOut});
	true ->
		PidsIn
	end;
new(N) ->
	if N > 0 ->
		PidsIn = [],
		PidsOut = lists:append([spawn(test, neuron, [[], 2, 2])], PidsIn),
		new({N-1, PidsOut});
	true ->
		io:fwrite("Unacceptable number of neurons!")
	end.

connect(InPid, OutPid) ->
	OutPid ! {connect, InPid}.
	
stop([]) ->
	io:fwrite("Network stopped!");
stop(Pids) ->
	[H|T] = Pids,
	H ! stop,
	stop(T).
	
