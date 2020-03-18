import numpy as np



def write_data_to_file(filename, data):

    vec = ''
    with open(filename, 'a') as f:
        if np.ndim(data) == 0:
            vec += str(data) + '\n'

        elif np.ndim(data) == 1:
            for di in data:
                vec += str(di)+' '
            vec += '\n'

        elif np.ndim(data) == 2:
            nrows, ncols = int(np.shape(data)[0]), int(np.shape(data)[1])
            for nrow in range(0, nrows):
                for ncol in range(0, ncols):
                    vec += str(data[nrow, ncol]) + ' '
                vec += '\n'
        else:
            raise Exception('can only handle up to 2-D arrays.')

        f.write(vec)

def approx_theta_E(ximg,yimg):

    dis = []
    xinds,yinds = [0,0,0,1,1,2],[1,2,3,2,3,3]

    for (i,j) in zip(xinds,yinds):

        dx,dy = ximg[i] - ximg[j], yimg[i] - yimg[j]
        dr = (dx**2+dy**2)**0.5
        dis.append(dr)
    dis = np.array(dis)

    greatest = np.argmax(dis)
    dr_greatest = dis[greatest]
    dis[greatest] = 0

    second_greatest = np.argmax(dis)
    dr_second = dis[second_greatest]

    return 0.5*(dr_greatest*dr_second)**0.5

def image_separation_vectors_quad(ximg, yimg):

    dr = lambda x1, x2, y1, y2: np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    ximg, yimg = np.array(ximg), np.array(yimg)
    d1 = dr(ximg[0], ximg[1:], yimg[0], yimg[1:])
    d2 = dr(ximg[1], [ximg[0], ximg[2], ximg[3]], yimg[1],
            [yimg[0], yimg[2], yimg[3]])
    d3 = dr(ximg[2], [ximg[0], ximg[1], ximg[3]], yimg[2],
            [yimg[0], yimg[1], yimg[3]])
    d4 = dr(ximg[3], [ximg[0], ximg[1], ximg[2]], yimg[3],
            [yimg[0], yimg[1], yimg[2]])
    idx1 = np.argmin(d1)
    idx2 = np.argmin(d2)
    idx3 = np.argmin(d3)
    idx4 = np.argmin(d4)

    x_2, x_3, x_4 = [ximg[0], ximg[2], ximg[3]], [ximg[0], ximg[1], ximg[3]], [ximg[0], ximg[1], ximg[2]]
    y_2, y_3, y_4 = [yimg[0], yimg[2], yimg[3]], [yimg[0], yimg[1], yimg[3]], [yimg[0], yimg[1], yimg[2]]

    theta1 = np.arctan((yimg[1:][idx1] - yimg[0])/(ximg[1:][idx1] - ximg[0]))
    theta2 = np.arctan((y_2[idx2] - yimg[1]) / (x_2[idx2] - ximg[1]))
    theta3 = np.arctan((y_3[idx3] - yimg[2]) / (x_3[idx3] - ximg[2]))
    theta4 = np.arctan((y_4[idx4] - yimg[3]) / (x_4[idx4] - ximg[3]))

    return np.array([np.min(d1), np.min(d2), np.min(d3), np.min(d4)]), np.array([theta1, theta2,
                                                              theta3, theta4])
