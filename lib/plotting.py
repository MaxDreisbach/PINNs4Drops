import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.interpolate import griddata
import pyvista

def plot_contour(opt, samples, preds, labels, plane_dim, name, type, sample_name, dataset_type):
    sample_x = samples[0, 0, :].detach().cpu().numpy()
    sample_y = samples[0, 1, :].detach().cpu().numpy()
    sample_z = samples[0, 2, :].detach().cpu().numpy()
    sample = np.vstack((sample_x, sample_y, sample_z)).T
    #sample = samples.detach().cpu().numpy()
    label = labels.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.01:0.01:1j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.99:8.01:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.01:0.01:1j]

    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_linear = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_linear[np.isnan(pred_interpn)]

    label_interpn = griddata(sample, label, (X,Y,Z), method='linear')
    label_linear = griddata(sample, label, (X,Y,Z), method='nearest')
    label_interpn[np.isnan(label_interpn)] = label_linear[np.isnan(label_interpn)]

    if plane_dim == 'x':
        var_plot = np.squeeze(pred_interpn[0, :, :])
        gt_plot = np.squeeze(label_interpn[0, :, :])
    if plane_dim == 'y':
        var_plot = np.squeeze(pred_interpn[:, 0, :].T)
        gt_plot = np.squeeze(label_interpn[:, 0, :].T)
    if plane_dim == 'z':
        var_plot = np.squeeze(pred_interpn[:, :, 0].T)
        gt_plot = np.squeeze(label_interpn[:, :, 0].T)

    err_plot = np.absolute(gt_plot - var_plot)

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 10)
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.5, 1.5, 10)
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-0.6, 3.0, 10)
        colormap = 'viridis'

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.25))
    p1 = axs[0].contourf(var_plot, levels=levels, cmap=colormap)
    p2 = axs[1].contourf(gt_plot, levels=levels, cmap=colormap)
    p3 = axs[2].contourf(err_plot, cmap=colormap)
    axs[0].contourf(var_plot, levels=levels, cmap=colormap)
    axs[1].contourf(gt_plot, levels=levels, cmap=colormap)
    axs[2].contourf(err_plot, cmap=colormap)

    axs[0].set_ylabel('y', fontsize=16)
    axs[0].set_xlabel('x', fontsize=16)
    axs[1].set_xlabel('x', fontsize=16)
    axs[2].set_xlabel('x', fontsize=16)


    axs[0].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[1].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[2].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)

    axs[0].set_title(r'{0}_pred'.format(name), fontsize=16, y=1, pad=20 )
    axs[1].set_title(r'{0}_gt'.format(name), fontsize=16, y=1, pad=20)
    axs[2].set_title(r'%s_{err}' %name, fontsize=16, y=1, pad=20)

    #plt.tight_layout(w_pad=4.5)
    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    axs[2].set_aspect('equal', adjustable="datalim")
    fig.colorbar(p1, ax=axs[0], location='bottom')
    fig.colorbar(p2, ax=axs[1], location='bottom')
    fig.colorbar(p3, ax=axs[2], location='bottom')
    filename = 'results/'+ dataset_type + '_' + sample_name + '_' + name + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()

def plot_contour_w_alpha(opt, samples, preds, alpha, labels, plane_dim, name, type, sample_name, dataset_type):
    sample_x = samples[0, 0, :].detach().cpu().numpy()
    sample_y = samples[0, 1, :].detach().cpu().numpy()
    sample_z = samples[0, 2, :].detach().cpu().numpy()
    sample = np.vstack((sample_x, sample_y, sample_z)).T
    #sample = samples.detach().cpu().numpy()
    label = labels.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.01:0.01:1j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.99:8.01:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.01:0.01:1j]


    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_linear = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_linear[np.isnan(pred_interpn)]

    label_interpn = griddata(sample, label, (X,Y,Z), method='linear')
    label_linear = griddata(sample, label, (X,Y,Z), method='nearest')
    label_interpn[np.isnan(label_interpn)] = label_linear[np.isnan(label_interpn)]

    alpha_interpn = griddata(sample, alpha, (X, Y, Z), method='linear')
    alpha_linear = griddata(sample, alpha, (X, Y, Z), method='nearest')
    alpha_interpn[np.isnan(alpha_interpn)] = alpha_linear[np.isnan(alpha_interpn)]

    if plane_dim == 'x':
        var_plot = np.squeeze(pred_interpn[0, :, :])
        gt_plot = np.squeeze(label_interpn[0, :, :])
        alpha_plot = np.squeeze(alpha_interpn[0, :, :])
    if plane_dim == 'y':
        var_plot = np.squeeze(pred_interpn[:, 0, :].T)
        gt_plot = np.squeeze(label_interpn[:, 0, :].T)
        alpha_plot = np.squeeze(alpha_interpn[:, 0, :].T)
    if plane_dim == 'z':
        var_plot = np.squeeze(pred_interpn[:, :, 0].T)
        gt_plot = np.squeeze(label_interpn[:, :, 0].T)
        alpha_plot = np.squeeze(alpha_interpn[:, :, 0].T)

    err_plot = np.absolute(gt_plot - var_plot)

    x, y = np.meshgrid(np.arange(opt.resolution)/opt.resolution, np.arange(opt.resolution)/opt.resolution)

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 100)
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.5, 1.5, 100)
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-0.4, 3.2, 100)
        colormap = 'viridis'

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.75))
    p1 = axs[0].contourf(x, y, var_plot, levels=levels, cmap=colormap)
    p2 = axs[1].contourf(x, y, gt_plot, levels=levels, cmap=colormap)
    p3 = axs[2].contourf(x, y, err_plot, 100, cmap=colormap)


    levels_alpha = np.linspace(0.5, 1.0, 2)
    a1 = axs[0].contour(x, y, alpha_plot, levels=levels_alpha, colors='k')
    a2 = axs[1].contour(x, y, alpha_plot, levels=levels_alpha, colors='k')
    a3 = axs[2].contour(x, y, alpha_plot, levels=levels_alpha, colors='k')

    axs[0].set_ylabel('$y$', fontsize=16)
    axs[0].set_xlabel('$x$', fontsize=16)
    axs[1].set_xlabel('$x$', fontsize=16)
    axs[2].set_xlabel('$x$', fontsize=16)

    x = np.arange(0.0, 1.0 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)
    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)
    axs[2].set_xticks(x)
    axs[2].set_yticks(y)


    axs[0].tick_params(axis ='both', which ='major', labelsize = 20)
    axs[1].tick_params(axis ='both', which ='major', labelsize = 20)
    axs[2].tick_params(axis ='both', which ='major', labelsize = 20)

    axs[0].set_title('$%s_{pred}$' %name, fontsize=20, y=1, pad=20)
    axs[1].set_title('$%s_{gt}$' %name, fontsize=20, y=1, pad=20)
    axs[2].set_title('$%s_{err}$' %name, fontsize=20, y=1, pad=20)

    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    axs[2].set_aspect('equal', adjustable="datalim")
    cbar1 = fig.colorbar(p1, ax=axs[0], location='bottom')
    cbar2 = fig.colorbar(p2, ax=axs[1], location='bottom')
    cbar3 = fig.colorbar(p3, ax=axs[2], location='bottom')
    cbar1.ax.tick_params(labelsize=20, which='major', width=1.5, length=6)
    cbar2.ax.tick_params(labelsize=20, which='major')
    cbar3.ax.tick_params(labelsize=20, which='major')
    cbar1.ax.locator_params(nbins=5)
    cbar2.ax.locator_params(nbins=5)
    cbar3.ax.locator_params(nbins=5)

    filename = 'results/'+ dataset_type + '_' + sample_name + '_' + name + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()

def plot_contour_w_alpha_res_gt(opt, samples, preds, alpha, labels, plane_dim, name, type, sample_name, dataset_type):
    sample_x = samples[0, 0, :].detach().cpu().numpy()
    sample_y = samples[0, 1, :].detach().cpu().numpy()
    sample_z = samples[0, 2, :].detach().cpu().numpy()
    sample = np.vstack((sample_x, sample_y, sample_z)).T
    #sample = samples.detach().cpu().numpy()
    label = labels.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.01:0.01:1j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.99:8.01:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.01:0.01:1j]

    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_linear = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_linear[np.isnan(pred_interpn)]

    label_interpn = griddata(sample, label, (X,Y,Z), method='linear')
    label_linear = griddata(sample, label, (X,Y,Z), method='nearest')
    label_interpn[np.isnan(label_interpn)] = label_linear[np.isnan(label_interpn)]

    alpha_interpn = griddata(sample, alpha, (X, Y, Z), method='linear')
    alpha_linear = griddata(sample, alpha, (X, Y, Z), method='nearest')
    alpha_interpn[np.isnan(alpha_interpn)] = alpha_linear[np.isnan(alpha_interpn)]

    if plane_dim == 'x':
        var_plot = np.squeeze(pred_interpn[0, :, :])
        gt_plot = np.squeeze(label_interpn[0, :, :])
        alpha_plot = np.squeeze(alpha_interpn[0, :, :])
    if plane_dim == 'y':
        var_plot = np.squeeze(pred_interpn[:, 0, :].T)
        gt_plot = np.squeeze(label_interpn[:, 0, :].T)
        alpha_plot = np.squeeze(alpha_interpn[:, 0, :].T)
    if plane_dim == 'z':
        var_plot = np.squeeze(pred_interpn[:, :, 0].T)
        gt_plot = np.squeeze(label_interpn[:, :, 0].T)
        alpha_plot = np.squeeze(alpha_interpn[:, :, 0].T)

    err_plot = np.absolute(gt_plot - var_plot)

    x, y = np.meshgrid(np.arange(opt.resolution)/opt.resolution, np.arange(opt.resolution)/opt.resolution)

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 100)
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.5, 1.5, 100)
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-0.4, 3.2, 100)
        colormap = 'viridis'

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    #plt.rcParams['figure.constrained_layout.use'] = True

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(10, 6.5))
    p1 = axs[0].contourf(x, y, var_plot, levels=levels, cmap=colormap)
    p2 = axs[1].contourf(x, y, gt_plot, levels=levels, cmap=colormap)


    levels_alpha = np.linspace(0.5, 1.0, 2)
    a1 = axs[0].contour(x, y, alpha_plot, levels=levels_alpha, colors='k')
    a2 = axs[1].contour(x, y, alpha_plot, levels=levels_alpha, colors='k')

    axs[0].set_ylabel('$y$', fontsize=16)
    axs[0].set_xlabel('$x$', fontsize=16)
    axs[1].set_xlabel('$x$', fontsize=16)

    x = np.arange(0.0, 1.0 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)
    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)

    axs[0].tick_params(axis ='both', which ='major', labelsize = 20, pad = 10)
    axs[1].tick_params(axis ='both', which ='major', labelsize = 20, pad = 10)

    axs[0].set_title('$%s_{pred}$' %name, fontsize=20, y=1, pad=20)
    axs[1].set_title('$%s_{gt}$' %name, fontsize=20, y=1, pad=20)

    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    cbar1 = fig.colorbar(p1, ax=axs[0], location='bottom')
    cbar2 = fig.colorbar(p2, ax=axs[1], location='bottom')
    cbar1.ax.tick_params(labelsize=20, which='major', width=1.5, length=6)
    cbar2.ax.tick_params(labelsize=20, which='major')
    cbar1.ax.locator_params(nbins=5)
    cbar2.ax.locator_params(nbins=5)
    plt.tight_layout(w_pad=4.5)

    filename = 'results/'+ dataset_type + '_' + sample_name + '_' + name + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()

def plot_contour_grid(pred, label, plane_dim, res, type):
    # interpolate point cloud to 2D-plane
    ind = res // 2

    if plane_dim == 'x':
        var_plot = pred[ind, :, :].T
        gt_plot = label[ind, :, :].T
    if plane_dim == 'y':
        var_plot = pred[:, ind, :].T
        gt_plot = label[:, ind, :].T
    if plane_dim == 'z':
        var_plot = pred[:, :, ind].T
        gt_plot = label[:, :, ind].T

    if type == 'alpha':
        v_min, v_max = 0, 1
        colormap = 'RdBu_r'
    if type == 'vel':
        v_min, v_max = -2, 2
        colormap = 'RdBu_r'
    if type == 'pres':
        v_min, v_max = -250, 1000
        colormap = 'viridis'

    err_plot = np.absolute(var_plot - gt_plot)

    fig, axs = plt.subplots(1, 3, figsize=(17,6))
    p1 = axs[0].contourf(var_plot, cmap=colormap)
    p2 = axs[1].contourf(gt_plot, cmap=colormap)
    p3 = axs[2].contourf(err_plot, cmap=colormap)
    axs[0].contourf(var_plot, cmap=colormap)
    axs[1].contourf(gt_plot, cmap=colormap)
    axs[2].contourf(err_plot, cmap=colormap)

    axs[0].set_ylabel('y', fontsize=16)
    axs[0].set_xlabel('x', fontsize=16)
    axs[1].set_xlabel('x', fontsize=16)
    axs[2].set_xlabel('x', fontsize=16)


    axs[0].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[1].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[2].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)

    axs[0].set_title(r'$U_{pred}$', fontsize=16, y=1, pad=20 )
    axs[1].set_title(r'$U_{GT}$', fontsize=16, y=1, pad=20)
    axs[2].set_title(r'$U_{err}$', fontsize=16, y=1, pad=20)

    x = np.arange(0, 512, 100)
    y = np.arange(0, 512, 100)
    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)
    axs[2].set_xticks(x)
    axs[2].set_yticks(y)
    plt.tight_layout(w_pad=4.5)
    plt.axis('equal')

    #cax = plt.axes([1.01, 0.15, 0.0091, 0.76])
    #cax = plt.axes()
    #cbar1 = plt.colorbar(p2)
    #cbar2 = plt.colorbar(p1)
    #cbar3 = plt.colorbar(p3)
    #cax.yaxis.tick_right()
    #cax.yaxis.set_label_position('right')
    #cbar1.set_label(r'$U$' , fontsize=12, fontweight='bold')
    #cbar1.ax.tick_params(labelsize=12)
    fig.colorbar(p1, ax=axs[0], location='bottom')
    fig.colorbar(p2, ax=axs[1], location='bottom')
    fig.colorbar(p3, ax=axs[2], location='bottom')
    plt.show()
    plt.savefig('modelA.pdf')


def plot_velocity_field(samples, preds, labels_u, labels_v, labels_w):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    sample_tensor = samples.detach().cpu().numpy()
    labels_u = labels_u.detach().cpu().numpy()
    labels_v = labels_v.detach().cpu().numpy()
    labels_w = labels_w.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()
    x = np.array(sample_tensor[0, 0, :])
    y = np.array(sample_tensor[0, 1, :])
    z = np.array(sample_tensor[0, 2, :])
    u = labels_u
    v = labels_v
    w = labels_w
    u_pred = pred[0, 1, :]
    u_pred = pred[0, 2, :]
    u_pred = pred[0, 3, :]

    # colorbar
    c = np.sqrt(np.abs(v) ** 2 + np.abs(u) ** 2 + np.abs(w) ** 2)
    c = (c.ravel() - c.min()) / c.ptp()
    # Repeat for each body line and two head lines
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    c = plt.cm.jet(c)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x[:1000], y[:1000], z[:1000], u[:1000], v[:1000], w[:1000], arrow_length_ratio=0.5, length=10.0)
    #qq = ax.quiver(x[:1000], y[:1000], z[:1000], u[:1000], v[:1000], w[:1000], colors=c, arrow_length_ratio=0.3, length=5.0)
    #plt.colorbar(qq, cmap=plt.cm.jet)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    plt.show()


def plot_iso_surface(opt, samples, preds, name, sample_name, dataset_type):
    sample_x = samples[0, 0, :].detach().cpu().numpy()
    sample_y = samples[0, 1, :].detach().cpu().numpy()
    sample_z = samples[0, 2, :].detach().cpu().numpy()
    sample = np.vstack((sample_x, sample_y, sample_z)).T
    pred = preds.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -128:128:grid_res]

    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_linear = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_linear[np.isnan(pred_interpn)]

    mesh = pyvista.StructuredGrid(X, Y, Z)
    mesh.point_data['values'] = pred_interpn.ravel(order='F')

    vmin = pred.min()
    vmax = pred.max()
    labels = dict(zlabel='Z', xlabel='X', ylabel='Y')
    contours = mesh.contour(np.linspace(vmin, vmax, 10))

    camera = pyvista.Camera()
    camera.position = (700.0, 100.0, 700.0)
    camera.focal_point = (5.0, 20.0, 5.0)

    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(mesh.outline(), color="k")
    p.add_mesh(contours, opacity=0.25, clim=[vmin, vmax])
    p.show_grid(**labels)
    p.add_axes(**labels)

    p.camera = camera
    filename = 'results/' + dataset_type + '_' + sample_name + '_' + name + '_pred_3d.svg'
    p.save_graphic(filename)
    #p.screenshot(filename, transparent_background=True)
    p.close()

    #meshname = 'results/' + dataset_type + '_' + sample_name + '_' + name + '_pred_3d.vtk'
    #mesh.save(meshname)

def gen_vtk_prediction(coords, preds, name, sample_name):
    X = coords[0, :, :, :]
    Y = coords[1, :, :, :]
    Z = coords[2, :, :, :]

    mesh = pyvista.StructuredGrid(X, Y, Z)
    mesh.point_data['values'] = preds.ravel(order='F')

    meshname = sample_name + '_' + name + '_pred_3d.vtk'
    mesh.save(meshname)

def plot_contour_eval(coords, opt, preds, alpha, plane_dim, name, type, sample_name, dataset_type):
    #X = coords[0, :, :, :]
    #Y = coords[1, :, :, :]
    #Z = coords[2, :, :, :]

    # interpolate point cloud to 2D-plane
    ind = opt.resolution // 2

    if plane_dim == 'x':
        var_plot = preds[ind, :, :].T
        alpha_plot = alpha[ind, :, :].T
    if plane_dim == 'y':
        var_plot = preds[:, ind, :].T
        alpha_plot = alpha[:, ind, :].T
    if plane_dim == 'z':
        var_plot = preds[:, :, ind].T
        alpha_plot = alpha[:, :, ind].T

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 100)
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.5, 1.5, 100)
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-0.4, 3.2, 100)
        colormap = 'viridis'

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    x, y = np.meshgrid(np.arange(opt.resolution) / opt.resolution, np.arange(opt.resolution) / opt.resolution)

    fig, axs = plt.subplots(figsize=(6.5, 6.5))
    #p1 = axs.contourf(x, y, var_plot, levels=levels, cmap=colormap)
    p1 = axs.contourf(x, y, var_plot, cmap=colormap)

    levels_alpha = np.linspace(0.5, 1.0, 2)
    a1 = axs.contour(x, y, alpha_plot, levels=levels_alpha, colors='k')

    axs.set_ylabel('$y$', fontsize=16)
    axs.set_xlabel('$x$', fontsize=16)

    x = np.arange(0.0, 1.0 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)
    axs.set_xticks(x)
    axs.set_yticks(y)
    axs.tick_params(axis='both', which='major', labelsize=20)
    axs.set_title('$%s_{pred}$' % name, fontsize=20, y=1, pad=20)

    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axs.set_aspect('equal', adjustable="datalim")
    cbar1 = fig.colorbar(p1, ax=axs, location='bottom')
    cbar1.ax.tick_params(labelsize=20, which='major', width=1.5, length=6)
    cbar1.ax.locator_params(nbins=5)

    filename = sample_name + '_' + name + '_' + plane_dim + '_pred.pdf'
    plt.savefig(filename)
    # plt.show()
