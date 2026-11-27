using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

public static class LinkPtz
{
    [ComImport, Guid("C6E13370-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IAMCameraControl
    {
        [PreserveSig] int GetRange(int property, out int min, out int max, out int step, out int defaultValue, out int flags);
        [PreserveSig] int Set(int property, int value, int flags);
        [PreserveSig] int Get(int property, out int value, out int flags);
    }

    public sealed class LensRange
    {
        public int min { get; set; }
        public int max { get; set; }
        public int step { get; set; }
        public int value { get; set; }
    }

    private const int Zoom = 3;
    private const int Focus = 6;
    private const int Manual = 2;
    private static IAMCameraControl lens;

    private static IAMCameraControl OpenLens()
    {
        if (lens == null) lens = Open(false) as IAMCameraControl;
        return lens;
    }

    private static LensRange ReadRange(int property)
    {
        var control = OpenLens();
        int min, max, step, defaultValue, capabilities, value, mode;
        if (control == null ||
            control.GetRange(property, out min, out max, out step, out defaultValue, out capabilities) < 0 ||
            (capabilities & Manual) == 0 || step <= 0 || max <= min ||
            control.Get(property, out value, out mode) < 0) return null;
        return new LensRange { min = min, max = max, step = step, value = value };
    }

    public static LensRange ProbeZoom() { return ReadRange(Zoom); }
    public static LensRange ProbeFocus() { return ReadRange(Focus); }

    public static LensRange AdjustZoom(int steps)
    {
        var range = ProbeZoom();
        if (range == null) throw new InvalidOperationException("Manual zoom is unavailable.");
        long lastStep = ((long)range.max - range.min) / range.step;
        long currentStep = (long)Math.Round(((double)range.value - range.min) / range.step);
        long nextStep = Math.Max(0L, Math.Min(lastStep, currentStep + steps));
        int value = (int)(range.min + nextStep * range.step);
        if (value != range.value)
            Marshal.ThrowExceptionForHR(lens.Set(Zoom, value, Manual));
        return ProbeZoom();
    }

    [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface ICreateDevEnum
    {
        [PreserveSig] int CreateClassEnumerator(ref Guid category, out IEnumMoniker devices, int flags);
    }

    [ComImport, Guid("28F54685-06FD-11D2-B27A-00A0C9223196"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IKsControl
    {
        [PreserveSig] int KsProperty([In, MarshalAs(UnmanagedType.LPArray)] byte[] property, int propertyLength,
            [In, Out, MarshalAs(UnmanagedType.LPArray)] byte[] data, int dataLength, out int returned);
    }

    [ComImport, Guid("720D4AC0-7533-11D0-A5D6-28DB04C10000"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IKsTopologyInfo
    {
        void NumCategories(out int count);
        void Category(int index, out Guid type);
        void NumConnections(out int count);
        void Connection(int index, IntPtr info);
        void NodeName(int index, IntPtr buffer, int length, out int actual);
        void NumNodes(out int count);
        void NodeType(int index, out Guid type);
        void CreateNodeInstance(int index, ref Guid iid, [MarshalAs(UnmanagedType.IUnknown)] out object instance);
    }

    [DllImport("ole32.dll")]
    private static extern int CreateBindCtx(int reserved, out IBindCtx context);

    private static readonly Guid CameraControl = new Guid("C6E13370-30AC-11D0-A18C-00A0C9118956");
    private const int PanTiltRelative = 17;
    private static IKsControl camera;
    private static int node = -1;
    private static int nodeCount;
    private static IKsTopologyInfo topology;
    private static Guid propertySet = CameraControl;
    private static int selector = PanTiltRelative;

    private static object Open(bool readTopology)
    {
        var enumerator = (ICreateDevEnum)Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86")));
        var category = new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
        IEnumMoniker devices;
        Marshal.ThrowExceptionForHR(enumerator.CreateClassEnumerator(ref category, out devices, 0));
        if (devices != null)
        {
            IBindCtx context;
            Marshal.ThrowExceptionForHR(CreateBindCtx(0, out context));
            var item = new IMoniker[1];
            while (devices.Next(1, item, IntPtr.Zero) == 0)
            {
                string path;
                item[0].GetDisplayName(context, null, out path);
                if (path.IndexOf("vid_2e1a&pid_4c04", StringComparison.OrdinalIgnoreCase) < 0) continue;
                var filterId = new Guid("56A86895-0AD4-11CE-B03A-0020AF0BA770");
                object filter;
                item[0].BindToObject(context, null, ref filterId, out filter);
                if (!readTopology) return filter;
                topology = filter as IKsTopologyInfo;
                nodeCount = 0;
                if (topology != null) topology.NumNodes(out nodeCount);
                return filter;
            }
        }
        throw new InvalidOperationException("Insta360 Link 2 (2E1A:4C04) was not found.");
    }

    private static void Write(byte[] buffer, int offset, int value)
    {
        Buffer.BlockCopy(BitConverter.GetBytes(value), 0, buffer, offset, 4);
    }

    private static byte[] Property(int operation, int size)
    {
        var property = new byte[size + (node < 0 ? 0 : 8)];
        Buffer.BlockCopy(propertySet.ToByteArray(), 0, property, 0, 16);
        Write(property, 16, selector);
        Write(property, 20, operation | (node < 0 ? 0 : 0x10000000));
        if (node >= 0) Write(property, 24, node);
        return property;
    }

    public static void Probe()
    {
        camera = (IKsControl)Open(true);
        for (node = 0; node < nodeCount; node++)
        {
            Guid type;
            topology.NodeType(node, out type);
            if (type != new Guid("941C7AC0-C559-11D0-8A2B-00A0C9255AC1")) continue;
            var query = Property(0x100, 24);
            Array.Clear(query, 0, 20);
            var sets = new byte[4096];
            int count;
            var hr = camera.KsProperty(query, query.Length, sets, sets.Length, out count);
            for (int offset = 0; hr >= 0 && offset + 16 <= count; offset += 16)
            {
                var guid = new byte[16];
                Buffer.BlockCopy(sets, offset, guid, 0, 16);
                propertySet = new Guid(guid);
                selector = 0x16;
                var property = Property(1, 24);
                var data = new byte[4];
                int length;
                if (camera.KsProperty(property, property.Length, data, data.Length, out length) >= 0 && length == 4) return;
            }
        }
        camera = null;
        throw new InvalidOperationException("Relative PTZ SET unavailable through Windows camera driver.");
    }

    private static bool SupportsSet(IKsControl control)
    {
        var property = Property(0x200, 24);
        var access = new byte[4];
        int returned;
        var hr = control.KsProperty(property, property.Length, access, access.Length, out returned);
        return hr >= 0 && returned >= 4 && (BitConverter.ToInt32(access, 0) & 2) != 0;
    }

    public static void Move(int pan, int tilt)
    {
        if (pan < -20 || pan > 20 || tilt < -20 || tilt > 20) throw new ArgumentOutOfRangeException();
        if (camera == null) Probe();
        var property = Property(2, 24);
        var data = new[] { unchecked((byte)Math.Sign(pan)), (byte)Math.Max(1, Math.Abs(pan)),
            unchecked((byte)Math.Sign(tilt)), (byte)Math.Max(1, Math.Abs(tilt)) };
        int returned;
        var hr = camera.KsProperty(property, property.Length, data, data.Length, out returned);
        if (hr < 0)
        {
            camera = null;
            throw new InvalidOperationException(String.Format("Relative PTZ SET failed (0x{0:X8}).", hr));
        }
    }
}
