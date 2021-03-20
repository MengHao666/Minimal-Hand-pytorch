import matplotlib.pyplot as plt


def plot3d(joints_,ax, title=None):
    joints = joints_.copy()
    ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], 'yo', label='keypoint')

    ax.plot(joints[:5, 0], joints[:5, 1],
             joints[:5, 2],
             'r',
             label='thumb')

    ax.plot(joints[[0, 5, 6, 7, 8, ], 0], joints[[0, 5, 6, 7, 8, ], 1],
             joints[[0, 5, 6, 7, 8, ], 2],
             'b',
             label='index')
    ax.plot(joints[[0, 9, 10, 11, 12, ], 0], joints[[0, 9, 10, 11, 12], 1],
             joints[[0, 9, 10, 11, 12], 2],
             'b',
             label='middle')
    ax.plot(joints[[0, 13, 14, 15, 16], 0], joints[[0, 13, 14, 15, 16], 1],
             joints[[0, 13, 14, 15, 16], 2],
             'b',
             label='ring')
    ax.plot(joints[[0, 17, 18, 19, 20], 0], joints[[0, 17, 18, 19, 20], 1],
             joints[[0, 17, 18, 19, 20], 2],
             'b',
             label='pinky')
    # snap convention
    ax.plot(joints[4][0], joints[4][1], joints[4][2], 'rD', label='thumb')
    ax.plot(joints[8][0], joints[8][1], joints[8][2], 'ro', label='index')
    ax.plot(joints[12][0], joints[12][1], joints[12][2], 'ro', label='middle')
    ax.plot(joints[16][0], joints[16][1], joints[16][2], 'ro', label='ring')
    ax.plot(joints[20][0], joints[20][1], joints[20][2], 'ro', label='pinky')
    # plt.plot(joints [1:, 0], joints [1:, 1], joints [1:, 2], 'o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xmin=-1.0,xmax=1.0)
    ax.set_ylim(ymin=-1.0,ymax=1.0)
    ax.set_zlim(zmin=-1.0,zmax=1.0)
    # plt.legend()
    # ax.view_init(330, 110)
    ax.view_init(-90, -90)
    return ax


def multi_plot3d(jointss_, title=None):
    jointss = jointss_.copy()
    fig = plt.figure(figsize=[50, 50])

    ax = fig.add_subplot(111, projection='3d')

    colors = ['b', 'r', "g"]

    for i in range(len(jointss)):
        joints = jointss[i]

        plt.plot(joints[:, 0], joints[:, 1], joints[:, 2], 'yo')

        plt.plot(joints[:5, 0], joints[:5, 1],
                 joints[:5, 2],
                 colors[i],
                 )

        plt.plot(joints[[0, 5, 6, 7, 8, ], 0], joints[[0, 5, 6, 7, 8, ], 1],
                 joints[[0, 5, 6, 7, 8, ], 2],
                 colors[i],
                 )
        plt.plot(joints[[0, 9, 10, 11, 12, ], 0], joints[[0, 9, 10, 11, 12], 1],
                 joints[[0, 9, 10, 11, 12], 2],
                 colors[i],
                 )
        plt.plot(joints[[0, 13, 14, 15, 16], 0], joints[[0, 13, 14, 15, 16], 1],
                 joints[[0, 13, 14, 15, 16], 2],
                 colors[i],
                 )
        plt.plot(joints[[0, 17, 18, 19, 20], 0], joints[[0, 17, 18, 19, 20], 1],
                 joints[[0, 17, 18, 19, 20], 2],
                 colors[i],
                 )

        #######
        # plt.plot(joints[:1, 0], joints[:1, 1],
        #          joints[:1, 2],
        #          colors[i],
        #          )
        #
        # plt.plot(joints[[0, 5,  ], 0], joints[[0, 5, ], 1],
        #          joints[[0, 5,  ], 2],
        #          colors[i],
        #          )
        # plt.plot(joints[[0, 9, ], 0], joints[[0, 9, ], 1],
        #          joints[[0, 9,], 2],
        #          colors[i],
        #          )
        # plt.plot(joints[[0, 13, ], 0], joints[[0, 13, ], 1],
        #          joints[[0, 13, ], 2],
        #          colors[i],
        #          )
        # plt.plot(joints[[0, 17, ], 0], joints[[0, 17, ], 1],
        #          joints[[0, 17, ], 2],
        #          colors[i],
        #          )

        # snap convention
        plt.plot(joints[4][0], joints[4][1], joints[4][2], 'rD')
        plt.plot(joints[8][0], joints[8][1], joints[8][2], 'ro', )
        plt.plot(joints[12][0], joints[12][1], joints[12][2], 'ro', )
        plt.plot(joints[16][0], joints[16][1], joints[16][2], 'ro', )
        plt.plot(joints[20][0], joints[20][1], joints[20][2], 'ro', )
        # plt.plot(joints [1:, 0], joints [1:, 1], joints [1:, 2], 'o')

        plt.title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend()
        # ax.view_init(330, 110)
        ax.view_init(-90, -90)

    if title:
        title_ = ""
        for  i in range(len(title)):
            title_ += "{}: {}   ".format(colors[i], title[i])

        ax.set_title(title_, fontsize=12, color='black')
    else:
        ax.set_title("None", fontsize=12, color='black')
    plt.show()
