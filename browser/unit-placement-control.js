
var Mode = {
    SelectUnit : 0,
    PlaceUnit : 1
};

var mode = Mode.SelectUnit;
var selectedUnit = null;

var viewBoxZoomedIn = true;

function hexMouseDownHandler(evt) {
    if (mode === Mode.PlaceUnit) {
        let hex = Map.hexIndex[this.id];
        selectedUnit.setHex(hex);
        SVGUnitSymbol.unmarkSelected();
        SVGUnitSymbol.moveSymbolToHex(selectedUnit.uniqueId, hex);
        mode = Mode.SelectUnit;
    }
}

function hexMouseOverHandler(evt) {
}

function edgeMouseOver(evt) {
}

function unitMouseDownHandler(evt) {
    if (mode === Mode.SelectUnit) {
        selectedUnit = Unit.unitIndex[this.id];
        SVGUnitSymbol.markSelected(this);
        mode = Mode.PlaceUnit;
    }
    else if (mode === Mode.PlaceUnit) {
        SVGUnitSymbol.unmarkSelected();
        // Switch locations
        let switchTarget = Unit.unitIndex[this.id];
        let hex_swap = switchTarget.hex;
        switchTarget.setHex(selectedUnit.hex);
        selectedUnit.setHex(hex_swap);
        SVGUnitSymbol.moveSymbolToHex(selectedUnit.uniqueId, selectedUnit.hex);
        SVGUnitSymbol.moveSymbolToHex(switchTarget.uniqueId, switchTarget.hex);
        mode = Mode.SelectUnit;
    }
}

function svgMouseDownHandler(evt) {
    if (evt.shiftKey) {
        const svg = this;
        const vbox = svg.viewBox.baseVal;
        if (viewBoxZoomedIn) { 
            // Zoom out       
            const bbox = SVG.svg.getBBox();
            const bbw = bbox.width + bbox.x;
            const bbh = bbox.height + bbox.y;
            if (vbox.width < bbw || vbox.height < bbh) {
                const c = Math.max( bbw/vbox.width, bbh/vbox.height );
                vbox.x = 0;
                vbox.y = 0;
                vbox.width *= c;
                vbox.height *= c;
            }
            viewBoxZoomedIn = false;
        }
        else {
            // Zoom in centered on mouse
            const bb = svg.getBoundingClientRect();
            x_frac = (evt.x - bb.x)/bb.width;
            y_frac = (evt.y - bb.y)/bb.height;
            xc_vb = vbox.x + x_frac * vbox.width;
            yc_vb = vbox.y + y_frac * vbox.height;
            vbox.x = xc_vb - SVGUtil.vbZoomedWidth/2;
            vbox.y = yc_vb - SVGUtil.vbZoomedHeight/2;
            vbox.width = SVGUtil.vbZoomedWidth;
            vbox.height = SVGUtil.vbZoomedHeight;
            viewBoxZoomedIn = true;
        }

    }
}
