import pm4py

if __name__ == "__main__":
    log = pm4py.read_xes('bpi-chall.xes')
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)
    pm4py.view_petri_net(net, initial_marking, final_marking, format="svg")

    process_tree = pm4py.discover_process_tree_inductive(log)
    bpmn_model = pm4py.convert_to_bpmn(process_tree)
    pm4py.view_bpmn(bpmn_model)