import numpy as np

def chain_code_func(axis_info, xidxst, yidxst, xidxed, yidxed):
    nrow, ncol = axis_info.shape
    numoftotalpoints = np.sum(axis_info == 1)
    flags = np.zeros((nrow, ncol), dtype=bool)
    xidxpre = xidxst
    yidxpre = yidxst
    contour_points = np.zeros((numoftotalpoints, 2))
    numofpoints = 1
    flags[yidxst, xidxst] = True
    contour_points[numofpoints - 1] = [yidxst, xidxst]
    chain_code = []

    while numofpoints < numoftotalpoints:
        if xidxst == xidxed and yidxst == yidxed:
            break

        yidx = np.clip(yidxst - 1, 0, nrow - 1)
        xidx = np.clip(xidxst, 0, ncol - 1)
        if axis_info[yidx, xidx] == 1 and not flags[yidx, xidx]:
            numofpoints += 1
            contour_points[numofpoints - 1] = [yidx, xidx]
            flags[yidx, xidx] = True
            yidxpre, xidxpre = yidxst, xidxst
            yidxst, xidxst = yidx, xidx
            chain_code.append(2)

        else:
            yidx = np.clip(yidxst, 0, nrow - 1)
            xidx = np.clip(xidxst + 1, 0, ncol - 1)
            if axis_info[yidx, xidx] == 1 and not flags[yidx, xidx]:
                numofpoints += 1
                contour_points[numofpoints - 1] = [yidx, xidx]
                flags[yidx, xidx] = True
                yidxpre, xidxpre = yidxst, xidxst
                yidxst, xidxst = yidx, xidx
                chain_code.append(0)

            else:
                yidx = np.clip(yidxst + 1, 0, nrow - 1)
                xidx = np.clip(xidxst, 0, ncol - 1)
                if axis_info[yidx, xidx] == 1 and not flags[yidx, xidx]:
                    numofpoints += 1
                    contour_points[numofpoints - 1] = [yidx, xidx]
                    flags[yidx, xidx] = True
                    yidxpre, xidxpre = yidxst, xidxst
                    yidxst, xidxst = yidx, xidx
                    chain_code.append(6)

                else:
                    yidx = np.clip(yidxst, 0, nrow - 1)
                    xidx = np.clip(xidxst - 1, 0, ncol - 1)
                    if axis_info[yidx, xidx] == 1 and not flags[yidx, xidx]:
                        numofpoints += 1
                        contour_points[numofpoints - 1] = [yidx, xidx]
                        flags[yidx, xidx] = True
                        yidxpre, xidxpre = yidxst, xidxst
                        yidxst, xidxst = yidx, xidx
                        chain_code.append(4)

                    else:
                        yidx = np.clip(yidxst + 1, 0, nrow - 1)
                        xidx = np.clip(xidxst + 1, 0, ncol - 1)
                        if axis_info[yidx, xidx] == 1 and not flags[yidx, xidx]:
                            numofpoints += 1
                            contour_points[numofpoints - 1] = [yidx, xidx]
                            flags[yidx, xidx] = True
                            yidxpre, xidxpre = yidxst, xidxst
                            yidxst, xidxst = yidx, xidx
                            chain_code.append(7)

                        else:
                            yidx = np.clip(yidxst - 1, 0, nrow - 1)
                            xidx = np.clip(xidxst + 1, 0, ncol - 1)
                            if axis_info[yidx, xidx] == 1 and not flags[yidx, xidx]:
                                numofpoints += 1
                                contour_points[numofpoints - 1] = [yidx, xidx]
                                flags[yidx, xidx] = True
                                yidxpre, xidxpre = yidxst, xidxst
                                yidxst, xidxst = yidx, xidx
                                chain_code.append(1)

                            else:
                                yidx = np.clip(yidxst + 1, 0, nrow - 1)
                                xidx = np.clip(xidxst - 1, 0, ncol - 1)
                                if axis_info[yidx, xidx] == 1 and not flags[yidx, xidx]:
                                    numofpoints += 1
                                    contour_points[numofpoints - 1] = [yidx, xidx]
                                    flags[yidx, xidx] = True
                                    yidxpre, xidxpre = yidxst, xidxst
                                    yidxst, xidxst = yidx, xidx
                                    chain_code.append(5)
                                else:
                                    yidx = np.clip(yidxst - 1, 0, nrow - 1)
                                    xidx = np.clip(xidxst - 1, 0, ncol - 1)
                                    if axis_info[yidx, xidx] == 1 and not flags[yidx, xidx]:
                                        numofpoints += 1
                                        contour_points[numofpoints - 1] = [yidx, xidx]
                                        flags[yidx, xidx] = True
                                        yidxpre, xidxpre = yidxst, xidxst
                                        yidxst, xidxst = yidx, xidx
                                        chain_code.append(3)
                                    else:
                                        break
    return chain_code
