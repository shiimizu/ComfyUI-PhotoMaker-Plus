import {app} from "/scripts/app.js"
import {api} from "/scripts/api.js";

app.registerExtension({
    name: "LoadRefImages",
    // imageWidget: undefined,
    // properties: undefined,
    // fileInput: undefined,
    // uploadWidget: undefined,

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "LoadRefImages"){
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            if (!this.properties) {
                this.properties = {};
            }

            this.serialize_widgets = true;

            this.imageWidget = this.widgets.find((w) => w.name === "image");
            this.imageWidget.callback = this.showImage.bind(this);
            console.log(this.imageWidget)

            const default_value = this.imageWidget.value;

            Object.defineProperty(this.imageWidget, "value", {
                set: function (value) {
                    this._real_value = value;
                },
                get: function () {
                    let value = "";
                    if (this._real_value) {
                        value = this._real_value;
                    } else {
                        return default_value;
                    }

                    if (value.filename) {
                        let real_value = value;
                        value = "";
                        if (real_value.subfolder) {
                            value = real_value.subfolder + "/";
                        }

                        value += real_value.filename;

                        if (real_value.type && real_value.type !== "input")
                            value += ` [${real_value.type}]`;
                    }
                    return value;
                }
            })

            this.fileInput = document.createElement("input");

            const accepted_files = {
                "LoadRefImages" : "image/png,image/webp,image/gif",
            }

            Object.assign(this.fileInput, {
                type: "file",
                accept: accepted_files[nodeData.name] ?? accepted_files["LoadRefImages"],
                style: "display: none",
                onchange: async () => {
                    if (this.fileInput.files.length) {
                        await this.uploadFile(this.fileInput.files[0], true);
                    }
                },
            });

            document.body.append(this.fileInput);

            this.uploadWidget = this.addWidget("button", "choose file to upload", "image", () => {
                this.fileInput.click();
            });
            this.uploadWidget.serialize = false;

            this.onDragOver = function (e) {
                if (e.dataTransfer && e.dataTransfer.items) {
                    const image = [...e.dataTransfer.items].find((f) => f.kind === "file");
                    return !!image;
                }

                return false;
            };

            this.onDragDrop = function (e) {
                console.log("onDragDrop called");
                let handled = false;
                for (const file of e.dataTransfer.files) {
                    if (file.type.startsWith("image/")) {
                        this.uploadFile(file, !handled); // Dont await these, any order is fine, only update on first one
                        handled = true;
                    }
                }

                return handled;
            };

            requestAnimationFrame(() => {
                if (this.imageWidget.value) {
                    this.showImage(this.imageWidget.value);
                }
            });
        }

        nodeType.prototype.showImage = function (name) {
            const img = new Image();
            img.onload = () => {
                this.imgs = [img];
                app.graph.setDirtyCanvas(true);
            };
            let folder_separator = name.lastIndexOf("/");
            let subfolder = "";
            if (folder_separator > -1) {
                subfolder = name.substring(0, folder_separator);
                name = name.substring(folder_separator + 1);
            }

            const imageSrc = `/view?filename=${name}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}`

            img.src = api.apiURL(imageSrc);
            console.log(img.src)
            this.setSizeForImage?.();
        }

        nodeType.prototype.uploadFile = async function (file, updateNode) {
            try {
                // Wrap file in formdata so it includes filename
                const body = new FormData();
                body.append("image", file);
                const resp = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body,
                });

                if (resp.status === 200) {
                    const data = await resp.json();
                    // Add the file as an option and update the widget value
                    if (!this.imageWidget.options.values.includes(data.name)) {
                        this.imageWidget.options.values.push(data.name);
                    }

                    if (updateNode) {
                        this.showImage(data.name);

                        this.imageWidget.value = data.name;
                    }
                } else {
                    alert(resp.status + " - " + resp.statusText);
                }
            } catch (error) {
                alert(error);
            }
        }
    },
});