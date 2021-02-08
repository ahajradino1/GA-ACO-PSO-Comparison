import matplotlib.pyplot as plt


def plot_scatter(result, best, f_range):
    plt.scatter(result[0], result[1])
    plt.plot([0], [0], marker='o', markersize=5, color="red")
    plt.plot([best[0]], [best[1]], marker='o', markersize=5, color="lime")
    plt.axis([f_range[0], f_range[1], f_range[0], f_range[1]])
    plt.show()


def plot_function_values(function_values, title):
    iterations = list(range(0, len(function_values)))

    fig, ax = plt.subplots()
    ax.plot(iterations, function_values)
    ax.set_title(title)
    ax.set_xticks(iterations, minor=False)

    for i in range(0, len(function_values)):
        ax.scatter(iterations[i], function_values[i], color='green', marker='o', s=20)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel('$Broj \ iteracije$')
    ax.set_ylabel('$f_{Broj \ iteracije}$')

    # make arrows
    ax.plot(1, 0, ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    plt.show()