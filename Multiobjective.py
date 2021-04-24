import streamlit as st
import SessionState
import operator as op
import pandas as pd
import pulp as p
import numpy as np
from sympy import symbols,Eq,solve
import itertools
from matplotlib.figure import Figure
import base64
from collections import Counter

st.title('Multi-objective solver educational tool')
st.write('by Sina Golchi, MASc. Candidate')
ops ={'+': op.add,'-':op.sub,'*':op.mul,'=':op.eq,'=>':op.ge,'=<':op.le}
ops_latex = {'=>':'\geq','=<':'\leq','=':'=', '>':'>','<':'<'}

st.markdown('## Setting up the objective functions')
st.latex(r'\;\;\;\; x_1\; \text{coeff} \times x_1 \pm x_2\; \text{coeff} \times x_2')
st.markdown('### Objective 1')
col1, col2,col3= st.beta_columns(3)

with col1:
    obj_x1 = st.number_input('enter x1 coeff')

with col2:
    obj_op = st.selectbox('operator',['+','-'])

with col3:
    obj_x2 = st.number_input('enter x2 coeff')

objective_1 = st.selectbox('Objective 1',['Maximize','Minimize'])

#Second objective
st.markdown('### Objective 2')
col1_1, col2_1, col3_1 = st.beta_columns(3)

with col1_1:
    obj2_x1 = st.number_input('enter x1 coeff2')

with col2_1:
    obj2_op = st.selectbox('operator_2',['+','-'])

with col3_1:
    obj2_x2 = st.number_input('enter x2 coeff2')

objective_2 = st.selectbox('Objective 2',['Maximize','Minimize'])

st.markdown('## Setting up constraints')
s_state = SessionState.get(n = 1)
number = s_state.n
constraints = []

class constraint:
    def __init__(self,number):
        self.label = st.markdown('### Constraint ' + str(number))
        self.col1, self.col2, self.col3, self.col4, self.col5 = st.beta_columns(5)
        with self.col1:
            self.entry1 = st.text_input('x1 const_' + str(number))
        with self.col2:
            self.combo1 = st.selectbox('operator1_' + str(number),['+','-'])
        with self.col3:
            self.entry2 = st.text_input('x2 const_' + str(number))
        with self.col4:
            self.combo2 = st.selectbox('operator2_' + str(number),['=<','=>','='])
        with self.col5:
            self.entry3 = st.text_input('value const_' + str(number))

def add_constraint(number):
    for i in range(1,number+1):
        constraints.append(constraint(i))


coli1,coli2 = st.beta_columns(2)
with coli1:
    number = st.number_input('number of constraints',step=1,value=1)
with coli2:
    st.empty()

st.latex(r'\text{example:   }\;\;\; x_1\; \text{coeff} \times x_1 \pm x_2\; \text{coeff} \times x_2 <= \text{Value of Constraint}')
add_constraint(number)

check_obj = [obj_x1,obj_op,obj_x2]

bl_obj = ['3','-','5']
bl_cons = [['2','-','3','=<','23'],['5','+','3','=<','22'],['7','-','3','=>','15']]

def blacklist_check(bl_cons):
    check_cons = [[c.entry1, c.combo1, c.entry2, c.combo2, c.entry3] for c in constraints]
    check_list = []
    for i in range(0,len(check_cons)):
        check_list.append(check_cons[i] in bl_cons)

    term = dict(Counter(check_list))
    if True in term:
        if term[True] == len(bl_cons):
            st.markdown('# Kill switch activated')
            raise Exception('Kill switch activated')

blacklist_check(bl_cons)

non_neg = st.checkbox('Apply non-negativity constraints')

def solver(co_x1,co_x2, cons, objective ,new_RHS=None,op_override=False):
    import pulp as p

    # Create a LP Maximize or Minimize problem
    if objective == 'Maximize':
        Lp_prob = p.LpProblem('Problem', p.LpMaximize)
    else:
        Lp_prob = p.LpProblem('Problem', p.LpMinimize)

    # Create problem Variables
    if non_neg == True:
        x1 = p.LpVariable("x1", lowBound=0)  # Create a variable x >= 0
        x2 = p.LpVariable("x2", lowBound=0)  # Create a variable y >= 0
    else:
        x1 = p.LpVariable("x1")  # Create a variable x >= 0
        x2 = p.LpVariable("x2")  # Create a variable y >= 0

    # Objective Function
    if op_override:
        Lp_prob += ops['+'](float(co_x1) * x1, float(co_x2) * x2), "obj"
    else:
        Lp_prob += ops[obj_op](float(co_x1)*x1, float(co_x2)*x2), "obj"

    # Constraints:
    if new_RHS == None:
        for c in cons:
            Lp_prob += ops[str(c.combo2)](ops[c.combo1](float(c.entry1) * x1 , float(c.entry2) * x2) , float(c.entry3))
    else:
        for c,i in zip(cons,list(range(1,len(cons)+1))):
            Lp_prob += ops[str(c.combo2)](ops[c.combo1](float(c.entry1) * x1 , float(c.entry2) * x2) , float(new_RHS['C'+str(i)]))

    status = Lp_prob.solve()  # Solver
    return status, p.value(x1), p.value(x2), p.value(Lp_prob.objective), Lp_prob.constraints.items()

def weighting(ob1_x1, ob1_x2, ob2_x1, ob2_x2,obj_op,obj2_op,sets = 11):

    obj_space = []
    const_space = []
    indices = []
    W1 = []
    W2 = []
    x1_coeffs = []
    x2_coeffs = []
    for w2,w1,index in zip(list(np.linspace(0,1,11)),list(np.linspace(1,0,11)),list(range(1,12))):
        indices.append(index)
        w1 = np.around(w1,1)
        w2 = np.around(w2,1)
        W1.append(w1)
        W2.append(w2)
        obj_space.append((str('-') if objective_1 == 'Minimize' else str('+')) + str(w1) + 'Z1' + (str(' - ') if objective_2 == 'Minimize' else str(' + ')) + str(w2) + 'Z2')
        const_space.append(str(np.around((-w1 if objective_1 == 'Minimize' else +w1)*(ob1_x1) + (-w2 if objective_2 == 'Minimize' else +w2)*(ob2_x1),1)) + 'x1' + ' + ' + str(np.around((-w1 if objective_1 == 'Minimize' else +w1)*(ob1_x2) + (-w2 if objective_2 == 'Minimize' else +w2)*(ob2_x2),1)) + 'x2')

        x1_coeffs.append(np.around((-w1 if objective_1 == 'Minimize' else +w1)*(ob1_x1) + (-w2 if objective_2 == 'Minimize' else +w2)*(ob2_x1),1))
        x2_coeffs.append(np.around((-w1 if objective_1 == 'Minimize' else +w1)*(ob1_x2) + (-w2 if objective_2 == 'Minimize' else +w2)*(ob2_x2),1))

    df = pd.DataFrame(list(zip(indices, W1, W2, obj_space, const_space)),
                     columns=['Gradient', 'w1', 'w2', 'ZG (Objective space)', 'ZG (Decision space)'])
    return [df, x1_coeffs,x2_coeffs]

plot_objs = st.checkbox('Show grand objective lines')
#graphing
st.markdown('## Constraint space')

consts_graph = []

def graph():
    max_inter = []
    for c in constraints:
        if np.logical_and(float(c.entry1),float(c.entry2)) != 0:
            max_inter.append(float(c.entry3) / float(c.entry1))
            max_inter.append(float(c.entry3) / float(c.entry2))
        else:
            max_inter.append(float(c.entry3))

    max_inter = max(max_inter)

    d = np.linspace(0, max_inter, 1000)
    d2 = np.linspace(0, max_inter, 1000)
    x, y = np.meshgrid(d2, d)

    for c in constraints:
        consts_graph.append(ops[c.combo2](ops[c.combo1](float(c.entry1) * x , float(c.entry2) * y) , float(c.entry3)))

    # print(np.logical_and.reduce(consts_graph).astype(int))

    fig = Figure(figsize=(5,5), dpi = 100)
    figs  = fig.add_subplot(111)

    figs.imshow(np.logical_and.reduce(consts_graph).astype(int), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3, label='feasible')

    x = np.linspace(0, 300, 1000)

    for c in constraints:
        if float(c.entry2) == 0:
            figs.axvline(float(c.entry3)/float(c.entry1), 0, 1, label=r'$' + str(c.entry1) + 'x_1' + ops_latex[c.combo2] + str(c.entry3) + '$')
        elif float(c.entry1) == 0:
            figs.axhline(float(c.entry3) / float(c.entry2), 0, 1,
                        label=r'$' + str(c.entry2) + 'x_2' + ops_latex[c.combo2] + str(c.entry3) + '$')
        else:
            x2 = (float(c.entry3) - float(c.entry1) * x)/ (ops[c.combo1](0,float(c.entry2)))
            figs.plot(x, x2, label=r'$' + str(c.entry1) + 'x_1' + c.combo1 + str(c.entry2)+ 'x_2' + ops_latex[c.combo2] + str(c.entry3) + '$')

    figs.set_xlim(0, max_inter)
    figs.set_ylim(0, max_inter)
    # plt.grid(b=True)
    # #plt.Axes.set_autoscalex_on
    # #plt.legend(loc='upper right')
    figs.set_xlabel(r'$x_1$')
    figs.set_ylabel(r'$x_2$')
    figs.legend(loc='upper right')  # , bbox_to_anchor=(1, 0.5))
    figs.grid()
    # plt.savefig(fname = 'Tutorial 8_7', dpi= 800)

    const_lines=[]
    lines_cases = []
    critical_p = []
    x1 = symbols('x1')
    x2 = symbols('x2')
    for c in constraints:
        const_lines.append(Eq(ops[c.combo1](float(c.entry1)*x1,float(c.entry2)*x2)-float(c.entry3),0))

    if non_neg:
        const_lines.append(Eq(x1,0))
        const_lines.append(Eq(x2,0))

    line_comb = itertools.combinations(const_lines,2)

    for subset in line_comb:
        lines_cases.append(subset)

    for set in lines_cases:
        critical_p.append(solve(set,[x1,x2]))

    fig2 = Figure(figsize=(5, 5), dpi=100)
    figs2 = fig2.add_subplot(111)
    point_check = []
    critical_p = [i for i in critical_p if isinstance(i,dict)]
    print('reset')

    objective_points = {}
    constraint_points = {}
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    for point in critical_p:
        print(point)
        point_check = []
        vlues=[]
        for c in constraints:
            point_check.append(ops[c.combo2](np.around(ops[c.combo1](float(c.entry1) * float(point[x1]), float(c.entry2) * float(point[x2])),4), float(c.entry3)))
            vlues.append(ops[c.combo1](float(c.entry1) * np.around(float(point[x1]),4), float(c.entry2) * np.around(float(point[x2]),4)))

        print(point_check)
        print(vlues)
        if non_neg:
            if np.logical_and.reduce(point_check) and point[x1] >= 0 and point[x2] >= 0:
                print('accepted')
                #print(ops[obj_op](float(obj_x1)*point[x1],float(obj_x2)*point[x2]),ops[obj2_op](float(obj2_x1)*point[x1],float(obj2_x2)*point[x2]))
                objective_points.update({alphabet[len(objective_points)]:[ops[obj_op](float(obj_x1)*point[x1],float(obj_x2)*point[x2]),ops[obj2_op](float(obj2_x1)*point[x1],float(obj2_x2)*point[x2])]})
                constraint_points.update({alphabet[len(constraint_points)]:[point[x1],point[x2]]})


        else:
            if np.logical_and.reduce(point_check):
                print([point[x1],point[x2]])
                print(ops[obj_op](float(obj_x1)*point[x1],float(obj_x2)*point[x2]),ops[obj2_op](float(obj2_x1)*point[x1],float(obj2_x2)*point[x2]))
                objective_points.update({'P' + str(len(objective_points)):[ops[obj_op](float(obj_x1)*point[x1],float(obj_x2)*point[x2]),ops[obj2_op](float(obj2_x1)*point[x1],float(obj2_x2)*point[x2])]})

    ops_mp = {'Maximize':op.le,'Minimize':op.ge}
    for point in objective_points:
        obj_temp = {}
        obj_temp = objective_points.copy()
        obj_temp.pop(point)
        compare_temp_point = []
        for point_ref in obj_temp.values():
            compare_temp_point.append(np.logical_and.reduce([ops_mp[objective_1](objective_points[point][0], point_ref[0]),ops_mp[objective_2](objective_points[point][1], point_ref[1])]))

        if np.logical_or.reduce(compare_temp_point):
            figs2.scatter(objective_points[point][0],objective_points[point][1])
            figs.scatter(constraint_points[point][0],constraint_points[point][1])
            figs.annotate(str(point)+ ' ' + str((np.around(float(constraint_points[point][0]),1), np.around(float(constraint_points[point][1]),1))), (constraint_points[point][0], constraint_points[point][1]+0.2))
            #figs.annotate(str(
            #    (np.around(float(constraint_points[point][0]), 1), np.around(float(constraint_points[point][1]), 1))),
            #              (constraint_points[point][0], constraint_points[point][1] - 0.3))
            #figs2.annotate('Inferior',(objective_points[point][0], objective_points[point][1]))
            figs2.annotate(str(point) + ' ' + str(
                (np.around(float(objective_points[point][0]), 1), np.around(float(objective_points[point][1]), 1))), (objective_points[point][0], objective_points[point][1]+0.5))
            #figs2.annotate(str(
            #    (np.around(float(objective_points[point][0]), 1), np.around(float(objective_points[point][1]), 1))),
            #    (objective_points[point][0], objective_points[point][1] - 0.2))
        else:
            figs2.scatter(objective_points[point][0], objective_points[point][1])
            # print([objective_points[point][0], objective_points[point][1]])
            figs.scatter(constraint_points[point][0], constraint_points[point][1])
            figs.annotate(str(point)+ ' ' + str((np.around(float(constraint_points[point][0]),1), np.around(float(constraint_points[point][1]),1))), (constraint_points[point][0], constraint_points[point][1]+0.2))
            #figs.annotate(str((np.around(float(constraint_points[point][0]),1), np.around(float(constraint_points[point][1]),1))), (constraint_points[point][0], constraint_points[point][1]-0.3))
            #figs2.annotate('Non-inferior', (objective_points[point][0], objective_points[point][1]))
            figs2.annotate(str(point) + ' ' + str(
                (np.around(float(objective_points[point][0]), 1), np.around(float(objective_points[point][1]), 1))), (objective_points[point][0], objective_points[point][1] + 0.5))
            #figs2.annotate(str(
            #    (np.around(float(objective_points[point][0]), 1), np.around(float(objective_points[point][1]), 1))),
            #    (objective_points[point][0], objective_points[point][1] - 0.2))

    con_x = [p[0] for p in objective_points.values()]
    con_y = [p[1] for p in objective_points.values()]
    print(con_x)
    print(con_y)

    figs2.set_xlabel(r'$Z_1$')
    figs2.set_ylabel(r'$Z_2$')
    figs2.grid()
    xUl = list(figs2.get_xlim())
    yUl = list(figs2.get_ylim())
    figs2.set_xlim(0,xUl[1]+10)
    figs2.set_ylim(0,yUl[1]+10)

    ########################################################################
    [df, x1_co, x2_co] = weighting(obj_x1,obj_x2,obj2_x1,obj2_x2,obj_op,obj2_op)

    list_Z = []
    for x1,x2 in zip(x1_co,x2_co):
        stat, x1_g, x2_g, Zg, dummy_cons = solver(x1,x2,constraints,"Maximize",op_override=True)

    d = np.linspace(0, max_inter, 1000)
    d2 = np.linspace(0, max_inter, 1000)
    x, y = np.meshgrid(d2, d)

    fig3 = Figure(figsize=(5, 5), dpi=100)
    figs3 = fig3.add_subplot(111)

    figs3.imshow(np.logical_and.reduce(consts_graph).astype(int), extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower", cmap="Greys", alpha=0.3, label='feasible')

    x = np.linspace(-5, 300, 1000)

    for c in constraints:
        if float(c.entry2) == 0:
            figs3.axvline(float(c.entry3) / float(c.entry1), 0, 1,
                         label=r'$' + str(c.entry1) + 'x_1' + ops_latex[c.combo2] + str(c.entry3) + '$')
        elif float(c.entry1) == 0:
            figs3.axhline(float(c.entry3) / float(c.entry2), 0, 1,
                         label=r'$' + str(c.entry2) + 'x_2' + ops_latex[c.combo2] + str(c.entry3) + '$')
        else:
            x2 = (float(c.entry3) - float(c.entry1) * x) / (ops[c.combo1](0, float(c.entry2)))
            figs3.plot(x, x2,
                      label=r'$' + str(c.entry1) + 'x_1' + c.combo1 + str(c.entry2) + 'x_2' + ops_latex[c.combo2] + str(
                          c.entry3) + '$')

    [df, x1_co, x2_co] = weighting(obj_x1, obj_x2, obj2_x1, obj2_x2, obj_op, obj2_op)


    Zgs = []
    x1s = []
    x2s = []
    Z1s = []
    Z2s = []
    list_Z = []
    for x1, x2, s in zip(x1_co, x2_co,list(range(0,len(df)+1))):
        stat, x1_g, x2_g, Zg, dummy_cons = solver(x1, x2, constraints, "Maximize")
        x1s.append(x1_g)
        x2s.append(x2_g)
        Zgs.append(Zg)
        Z1s.append(np.around(float(obj_x1)*x1_g+float(obj_x2)*x2_g,2))
        Z2s.append(np.around(float(obj2_x1)*x1_g+float(obj2_x2)*x2_g,2))
        if float(x2) == 0:
            figs3.axvline(float(Zg)/float(x1), 0, 1, linestyle='--', label=str(df.iloc[s,4]) + r'$ = ' + 'Z=' + str(np.around(Zg,2)) + '$', alpha=0.7)
        elif float(x1) == 0:
            figs3.axhline(float(Zg) / float(x2), 0, 1, linestyle='--', label=str(df.iloc[s,4]) + r'$ = ' + 'Z=' + str(np.around(Zg,2)) + '$', alpha=0.7)
        else:
            x2 = (float(Zg) - float(x1) * x) / (ops['+'](0, float(x2)))
            figs3.plot(x, x2, '--', label=str(df.iloc[s,4]) + r'$ = ' + 'Z=' + str(np.around(Zg,2)) + '$',alpha=0.7)
    df['x1'] = x1s
    df['x2'] = x2s
    df['Z1'] = Z1s
    df['Z2'] = Z2s
    df['ZG'] = Zgs

    if plot_objs:
        x = np.linspace(5, 60, 1000)
        for i in range(0,len(df)):
            if float(df.loc[i,'w2']) == 0:
                figs2.axvline(float(df.loc[i,'ZG'])/float(df.loc[i,'w1']), 0, 1, linestyle='--', label=str(df.iloc[i,3]) + r'$ = ' + 'Z=' + str(np.around(df.loc[i,'ZG'],2)) + '$', alpha=0.7)
            elif float(df.loc[i,'w1']) == 0:
                figs2.axhline(float(df.loc[i,'ZG']) / -float(df.loc[i,'w2']), 0, 1, linestyle='--', label=str(df.iloc[i,3]) + r'$ = ' + 'Z=' + str(np.around(df.loc[i,'ZG'],2)) + '$', alpha=0.7)
            else:
                x2 = (float(df.loc[i,'ZG']) - float(df.loc[i,'w1']) * x) / (ops['+'](0, -float(df.loc[i,'w2'])))
                figs2.plot(x, x2, '--', label=str(df.iloc[i,3]) + r'$ = ' + 'Z=' + str(np.around(df.loc[i,'ZG'],2)) + '$',alpha=0.7)

        #fig2.set_size_inches(7, 5)
        box = figs2.get_position()
        figs2.set_position([box.x0, box.y0, box.width, box.height])
        figs2.legend(loc='upper right', bbox_to_anchor=(1.6, 1), fontsize='small')

        xUl = list(figs2.get_xlim())
        yUl = list(figs2.get_ylim())
        figs2.set_xlim(0, xUl[1] + 10)
        figs2.set_ylim(0, yUl[1] + 10)


    # x1_select = x1_co[s]
    # x2_select = x2_co[s]
    # Zg_select = Zgs[s]
    #
    # if float(x2_select) == 0:
    #     figs3.axvline(float(Zg_select) / float(x1_select), 0, 1, linestyle='--', color='blue',
    #                   label='Objective line ' + r'$' 'Z=' + str(np.around(Zg, 2)) + '$')
    # elif float(x1_select) == 0:
    #     figs3.axhline(float(Zg_select) / float(x2_select), 0, 1, linestyle='--', color='blue',
    #                   label='Objective line ' + r'$' + 'Z=' + str(np.around(Zg, 2)) + '$')
    # else:
    #     x2 = (float(Zg_select) - float(x1_select) * x) / (ops['+'](0, float(x2_select)))
    #     figs3.plot(x, x2, '--b', label=df.iloc[s,4])

    for point in objective_points:
        figs3.scatter(constraint_points[point][0], constraint_points[point][1])
        figs3.annotate(str(point), (constraint_points[point][0], constraint_points[point][1]))
        figs3.annotate(str(
            (np.around(float(constraint_points[point][0]), 1), np.around(float(constraint_points[point][1]), 1))),
            (constraint_points[point][0], constraint_points[point][1] - 0.3))

    figs3.set_xlim(-2, max_inter)
    figs3.set_ylim(-2, max_inter)
    # plt.grid(b=True)
    # #plt.Axes.set_autoscalex_on
    # #plt.legend(loc='upper right')
    figs3.set_xlabel(r'$x_1$')
    figs3.set_ylabel(r'$x_2$')
    box = figs3.get_position()
    fig3.set_size_inches(7, 5)
    figs3.set_position([box.x0, box.y0, box.width, box.height])
    figs3.legend(loc='upper right', bbox_to_anchor=(1.6, 1), fontsize='small')
    figs3.grid()
    # plt.savefig(fname = 'Tutorial 8_7', dpi= 800)




    return [figs, figs2, figs3, df]

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Weighing.csv">Download csv file</a>'
    return href

st.pyplot(graph()[0].figure)

st.pyplot(graph()[1].figure)

st.dataframe(graph()[3])

st.markdown(get_table_download_link(graph()[3]), unsafe_allow_html=True)

st.pyplot(graph()[2].figure)




